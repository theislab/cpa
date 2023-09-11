from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial

from scvi.nn import FCLayers
from torch.distributions import Normal
from typing import Optional


class _REGISTRY_KEYS:
    X_KEY: str = "X"
    X_CTRL_KEY: str = None
    BATCH_KEY: str = None
    CATEGORY_KEY: str = "cpa_category"
    PERTURBATION_KEY: str = None
    PERTURBATION_DOSAGE_KEY: str = None
    PERTURBATIONS: str = "perts"
    PERTURBATIONS_DOSAGES: str = "perts_doses"
    SIZE_FACTOR_KEY: str = "size_factor"
    CAT_COV_KEYS: List[str] = []
    MAX_COMB_LENGTH: int = 2
    CONTROL_KEY: str = None
    DEG_MASK: str = None
    DEG_MASK_R2: str = None
    PADDING_IDX: int = 0


CPA_REGISTRY_KEYS = _REGISTRY_KEYS()


class VanillaEncoder(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
            n_cat_list,
            use_layer_norm=True,
            use_batch_norm=False,
            output_activation: str = 'linear',
            dropout_rate: float = 0.1,
            activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.n_output = n_output
        self.output_activation = output_activation

        self.network = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
        )
        self.z = nn.Linear(n_hidden, n_output)

    def forward(self, inputs, *cat_list):
        if self.output_activation == 'linear':
            z = self.z(self.network(inputs, *cat_list))
        elif self.output_activation == 'relu':
            z = F.relu(self.z(self.network(inputs, *cat_list)))
        else:
            raise ValueError(f'Unknown output activation: {self.output_activation}')
        return z


class GeneralizedSigmoid(nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, n_drugs, non_linearity='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.non_linearity = non_linearity
        self.n_drugs = n_drugs

        self.beta = torch.nn.Parameter(
            torch.ones(1, n_drugs),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, n_drugs),
            requires_grad=True
        )

        self.vmap = None

    def forward(self, x, y):
        """
            Parameters
            ----------
            x: (batch_size, max_comb_len)
            y: (batch_size, max_comb_len)
        """
        y = y.long()
        if self.non_linearity == 'logsigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (x * beta + bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.non_linearity == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class PerturbationNetwork(nn.Module):
    def __init__(self,
                 n_perts,
                 n_latent,
                 doser_type='logsigm',
                 n_hidden=None,
                 n_layers=None,
                 dropout_rate: float = 0.0,
                 drug_embeddings=None,):
        super().__init__()
        self.n_latent = n_latent
        
        if drug_embeddings is not None:
            self.pert_embedding = drug_embeddings
            self.pert_transformation = nn.Linear(drug_embeddings.embedding_dim, n_latent)
            self.use_rdkit = True
        else:
            self.use_rdkit = False
            self.pert_embedding = nn.Embedding(n_perts, n_latent, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
            
        self.doser_type = doser_type
        if self.doser_type == 'mlp':
            self.dosers = nn.ModuleList()
            for _ in range(n_perts):
                self.dosers.append(
                    FCLayers(
                        n_in=1,
                        n_out=1,
                        n_hidden=n_hidden,
                        n_layers=n_layers,
                        use_batch_norm=False,
                        use_layer_norm=True,
                        dropout_rate=dropout_rate
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(n_perts, non_linearity=self.doser_type)

    def forward(self, perts, dosages):
        """
            perts: (batch_size, max_comb_len)
            dosages: (batch_size, max_comb_len)
        """
        bs, max_comb_len = perts.shape
        perts = perts.long()
        scaled_dosages = self.dosers(dosages, perts)  # (batch_size, max_comb_len)

        drug_embeddings = self.pert_embedding(perts)  # (batch_size, max_comb_len, n_drug_emb_dim)

        if self.use_rdkit:
            drug_embeddings = self.pert_transformation(drug_embeddings.view(bs * max_comb_len, -1)).view(bs, max_comb_len, -1)

        z_drugs = torch.einsum('bm,bme->bme', [scaled_dosages, drug_embeddings])  # (batch_size, n_latent)

        z_drugs = torch.einsum('bmn,bm->bmn', z_drugs, (perts != CPA_REGISTRY_KEYS.PADDING_IDX).int()).sum(dim=1)  # mask single perts

        return z_drugs # (batch_size, n_latent)

class FocalLoss(nn.Module):
    """ Inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if len(y_true) == 0:
            return torch.tensor(0.)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(y_pred, dim=-1)
        ce = self.nll_loss(log_p, y_true)

        # get true class column from each row
        all_rows = torch.arange(len(y_pred))
        log_pt = log_p[all_rows, y_true]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss