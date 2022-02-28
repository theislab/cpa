import torch
import torch.nn as nn
import torch.nn.functional as F

from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._ann_dataloader import AnnDataLoader, BatchSampler
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi import settings

from typing import Optional, Union

from anndata import AnnData

from scvi.distributions import NegativeBinomial
from scvi.nn import FCLayers
from torch.distributions import Normal


class _CE_CONSTANTS:
    X_KEY = "X"
    PERTURBATIONS = "drugs_doses"
    DRUG_KEY = None
    DOSE_KEY = None
    CONTROL_KEY = None
    COVARS_KEYS = []
    # C_KEY = "covariates"
    # CAT_COVS_KEY = "cat_covs"
    # CONT_COVS_KEY = "cont_covs"
    # BATCH_KEY = "batch_indices"
    # LOCAL_L_MEAN_KEY = "local_l_mean"
    # LOCAL_L_VAR_KEY = "local_l_var"
    # LABELS_KEY = "labels"
    # PROTEIN_EXP_KEY = "protein_expression"


class DecoderNB(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
            use_layer_norm=True,
            use_batch_norm=False,
    ):
        super().__init__()
        self.hidd = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_output,
                n_layers=n_layers,
                n_hidden=n_hidden,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
            ),
            nn.Softmax(-1),
        )

    def forward(self, inputs, library, px_r):
        px_scale = self.hidd(inputs)
        px_rate = library.exp() * px_scale
        return NegativeBinomial(mu=px_rate, theta=px_r.exp())


class SimpleEncoder(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
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
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
        )
        self.z = nn.Linear(n_hidden, n_output)

    def forward(self, inputs):
        z = self.z(self.network(inputs))
        return z


class DecoderGauss(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
            use_layer_norm=True,
            use_batch_norm=False,
            output_activation: str = 'linear',
            dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_output = n_output
        self.output_activation = output_activation

        self.network = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation_fn=nn.ReLU,
        )
        self.mean = nn.Linear(n_hidden, n_output, bias=False)
        self.var = nn.Linear(n_hidden, n_output, bias=False)

    def forward(self, inputs):
        x = self.network(inputs)
        locs = self.mean(x)
        var_ = self.var(x)
        if self.output_activation == 'relu':
            locs = F.relu(locs)

        variances = var_.exp().add(1).log().add(1e-3)
        return Normal(loc=locs, scale=variances.sqrt())
        # return locs, variances


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

    def forward(self, x, y=None):
        """
            Parameters
            ----------
            x: (batch_size, n_drugs) or (batch_size, )
                Doses matrix
            y: (batch_size, )
        """
        if self.non_linearity == 'logsigm':
            if y is None:
                c0 = self.bias.sigmoid()
                return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
            else:
                bias = self.bias[0][y]
                beta = self.beta[0][y]
                c0 = bias.sigmoid()
                return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            if y is None:
                c0 = self.bias.sigmoid()
                return (x * self.beta + self.bias).sigmoid() - c0
            else:
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


class DrugNetwork(nn.Module):
    def __init__(self, n_drugs,
                 n_latent,
                 doser_type='logsigm',
                 n_hidden=None,
                 n_layers=None,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.n_latent = n_latent
        self.drug_embedding = nn.Embedding(n_drugs, n_latent)
        self.doser_type = doser_type
        if self.doser_type == 'mlp':
            self.dosers = nn.ModuleList()
            for _ in range(n_drugs):
                self.dosers.append(
                    FCLayers(
                        n_in=1,
                        n_out=1,
                        n_hidden=n_hidden,
                        n_layers=n_layers,
                        use_batch_norm=False,
                        use_layer_norm=False,
                        dropout_rate=dropout_rate
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(n_drugs, non_linearity=self.doser_type)

    def forward(self, drugs, doses=None):
        """
            drugs: (batch_size, n_drugs) if combinatorial else (batch_size, )
                OneHot multiplied by doses if combinatorial is True
        """
        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embedding.weight
        else:
            if doses is not None:
                drugs = drugs.long().view(-1, )
                doses = doses.float().view(-1, )
                scaled_dosages = self.dosers(doses, drugs)
                drug_embeddings = self.drug_embedding(drugs)
                return torch.einsum('b,be->be', [scaled_dosages, drug_embeddings])
            else:
                return self.dosers(drugs) @ self.drug_embedding.weight


class ManualDataSplitter(DataSplitter):
    """Manual train validation test splitter"""

    def __init__(
            self,
            adata: AnnData,
            train_idx,
            val_idx,
            test_idx,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.val_idx = val_idx
        self.train_idx = train_idx
        self.test_idx = test_idx

    def setup(self, stage: Optional[str] = None):
        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            if len(self.val_idx < 5000):
                data_loader_kwargs.update({'batch_size': len(self.val_idx)})
            else:
                data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
