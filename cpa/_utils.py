import torch
import torch.nn as nn
import torch.nn.functional as F


from scvi.distributions import NegativeBinomial
from scvi.nn import FCLayers


class _CE_CONSTANTS:
    X_KEY = "X"
    PERTURBATIONS = "drugs_doses"
    C_KEY = "covariates"
    CAT_COVS_KEY = "cat_covs"
    CONT_COVS_KEY = "cont_covs"
    BATCH_KEY = "batch_indices"
    LOCAL_L_MEAN_KEY = "local_l_mean"
    LOCAL_L_VAR_KEY = "local_l_var"
    LABELS_KEY = "labels"
    PROTEIN_EXP_KEY = "protein_expression"


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


class DecoderGauss(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_layers,
        use_layer_norm=True,
        use_batch_norm=False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_output = n_output

        self.network = FCLayers(
            n_in=n_input,
            n_out=n_output * 2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )

    def forward(self, inputs):
        x = self.network(inputs)
        locs = x[:, :self.n_output]
        var_ = x[:, self.n_output:]
        # variances = self.var_(hidd_)
        # TODO: Check Normal Distribution
        variances = var_.exp().add(1).log().add(1e-3)
        # return Normal(loc=locs, scale=variances.sqrt())
        return locs, variances


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
        self.beta = torch.nn.Parameter(
            torch.ones(1, n_drugs),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, n_drugs),
            requires_grad=True
        )

    def forward(self, x):
        """
            Parameters
            ----------
            x: (batch_size, n_drugs)
                Doses matrix 
        """
        if self.non_linearity == 'logsigm':
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == 'sigm':
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

    def forward(self, drugs):
        """
            drugs: (batch_size, n_drugs)
                OneHot multiplied by doses
        """
        if self.doser_type == 'mlp':
            doses = []
            for d in range(drugs.size(1)):
                this_drug = drugs[:, d].view(-1, 1)
                doses.append(self.dosers[d](this_drug).sigmoid() * this_drug.gt(0))
            return torch.cat(doses, 1) @ self.drug_embedding.weight
        else:
            return self.dosers(drugs) @ self.drug_embedding.weight

        