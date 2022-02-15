from sklearn.metrics import r2_score, make_scorer, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


import torch
import torch.distributions as db
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.nn.functional import one_hot

from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, FCLayers

from ._utils import _CE_CONSTANTS, DecoderGauss, DecoderNB, DrugNetwork

import numpy as np


class CPAModule(BaseModuleClass):
    """
    CPA module using Gaussian/NegativeBinomial Likelihood

    Parameters
    ----------
        n_genes: int
        n_treatments: int
        covars_to_ncovars: dict
            Dictionary of covariates with keys as each covariate name and values as 
                number of unique values of the corresponding covariate
        n_latent: int
            Latent Dimension
        loss_ae: str
            Autoencoder loss (either "gauss" or "nb")
        doser: str
            # TODO: What is this
        autoencoder_width: int
        autoencoder_depth: int
        use_batch_norm: bool
        use_layer_norm: bool
        variational: bool
    """
    def __init__(self,
                 n_genes: int,
                 n_drugs: int,
                 covars_to_ncovars: dict,
                 n_latent: int = 256,
                 loss_ae="gauss",
                 doser_type="linear",
                 autoencoder_width=256,
                 autoencoder_depth=2,
                 adversary_width=128,
                 adversary_depth=3,
                 dosers_width: int = 64,
                 dosers_depth: int = 2,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 variational: bool = False,
                 seed: int = 0,
                 ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_genes = n_genes
        self.n_drugs = n_drugs
        self.n_latent = n_latent
        self.loss_ae = loss_ae
        self.doser_type = doser_type
        self.ae_width = autoencoder_width
        self.ae_depth = autoencoder_depth
        self.dosers_width = dosers_width
        self.dosers_depth = dosers_depth
        self.adversary_width = adversary_width
        self.adversary_depth = adversary_depth
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.variational = variational

        self.covars_to_ncovars = covars_to_ncovars

        self.variational = variational
        if variational:
            self.encoder = Encoder(
                n_genes,
                n_latent,
                var_activation=nn.Softplus(),
                n_hidden=autoencoder_width, 
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=0.0,
            )
        else:
            self.encoder = FCLayers(
                n_in=n_genes,
                n_out=n_latent,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=0.0,
            )
        
        if self.loss_ae == 'nb':
            self.l_encoder = FCLayers(
                n_in=n_genes,
                n_out=1,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
            )

        # Decoder components
        if loss_ae in ["gauss", 'mse']:
            self.decoder = DecoderGauss(
                n_input=n_latent,
                n_output=n_genes,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=0.0,
            )
        elif loss_ae == 'nb':
            self.px_r = torch.nn.Parameter(torch.randn(n_genes))
            self.decoder = DecoderNB(
                n_input=n_latent,
                n_output=n_genes,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
            )
        
        else:
            raise Exception('Invalid Loss function for Autoencoder')

        # Embeddings
        # 1. Drug Network
        self.drug_network = DrugNetwork(n_drugs=self.n_drugs, 
                                        n_latent=self.n_latent, 
                                        doser_type=self.doser_type, 
                                        n_hidden=self.dosers_width, 
                                        n_layers=self.dosers_depth,
                                        dropout_rate=0.0
                                        )

        self.drugs_classifier = FCLayers(
            n_in=n_latent, 
            n_out=n_drugs,
            n_hidden=self.adversary_width,
            n_layers=self.adversary_depth,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=0.0,
        )

        # 2. Covariates Embedding
        self.covars_embedding = nn.ModuleDict(
            {
                key: torch.nn.Embedding(n_unique_cov_values, n_latent)
                for key, n_unique_cov_values in self.covars_to_ncovars.items()
            }
        )

        self.covars_classifiers = nn.ModuleDict(
            {
                key: FCLayers(n_in=n_latent, 
                              n_out=n_unique_cov_values,
                              n_hidden=self.adversary_width,
                              n_layers=self.adversary_depth,
                              use_batch_norm=use_batch_norm,
                              use_layer_norm=use_layer_norm,
                              dropout_rate=0.0)
                for key, n_unique_cov_values in self.covars_to_ncovars.items()
            }
        )

        self.adv_loss_covariates = {covar: nn.CrossEntropyLoss() for covar in self.covars_to_ncovars.keys()}
        self.adv_loss_drugs = nn.BCEWithLogitsLoss()

    def _get_inference_input(self, tensors):
        x = tensors[_CE_CONSTANTS.X_KEY] # batch_size, n_genes
        drugs_doses = tensors[_CE_CONSTANTS.PERTURBATIONS] # batch_size, n_drugs
        
        covars_dict = dict()
        # covars_onehot_dict = dict()
        for covar, n_covars in self.covars_to_ncovars.items():
            encoded_covars = tensors[covar] # (batch_size,)
            covars_dict[covar] = encoded_covars
            # if covar in self.covars_to_ncovars.keys():
            # val_oh = one_hot(encoded_covars.long().squeeze(), num_classes=n_covars)
            # else:
            #     val_oh = val
            # covars_onehot_dict[covar] = val_oh
        
        input_dict = dict(
            genes=x,
            drugs_doses=drugs_doses,
            covars_dict=covars_dict,
            # covars_onehot_dict=covars_onehot_dict,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        genes,
        drugs_doses,
        covars_dict,
    ):
        # x_ = torch.log1p(x)
        batch_size = genes.shape[0]
        x_ = genes
        if self.variational:
            qz_m, qz_v, latent_basal = self.encoder(x_)
            dist_qzbasal = db.Normal(qz_m, qz_v.sqrt())
        else:
            dist_qzbasal = None
            latent_basal = self.encoder(x_)
        
        if self.loss_ae == 'nb':
            library = self.l_encoder(x_)
        else:
            library = None

        latent_covariates = []
        for covar, _ in self.covars_to_ncovars.items():
            latent_covar_i = self.covars_embedding[covar](covars_dict[covar].long())
            latent_covar_i.view(batch_size, self.n_latent) # batch_size, n_latent
            latent_covariates.append(latent_covar_i)
        latent_covariates = torch.cat(latent_covariates, 0).sum(0)  # Summing all covariates representations
        latent_treatment = self.drug_network(drugs_doses)
        latent = latent_basal + latent_covariates + latent_treatment

        return dict(
            latent=latent,
            latent_basal=latent_basal,
            dist_qz=dist_qzbasal,
            library=library,
            covars_dict=covars_dict,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        input_dict = {}

        latent = inference_outputs["latent"]
        latent_basal = inference_outputs['latent_basal']
        if self.loss_ae == 'nb':
            library = inference_outputs["library"]
            input_dict["library"] = library

        covars_dict = dict()
        covars_onehot_dict = dict()
        for covar, n_covars in self.covars_to_ncovars.items():
            val = tensors[covar]
            covars_dict[covar] = val
            # if covar in self.covars_to_ncovars.keys():
            val_oh = one_hot(val.long().squeeze(), num_classes=n_covars)
            # else:
                # val_oh = val
            covars_onehot_dict[covar] = val_oh

        input_dict['latent'] = latent
        input_dict['latent_basal'] = latent_basal
        return input_dict

    @auto_move_data
    def generative(
        self,
        latent,
        latent_basal,
    ):
        drugs_pred = self.drugs_classifier(latent_basal)

        covars_pred = {}
        for covar in self.covars_to_ncovars.keys():
            covar_pred = self.covars_classifiers[covar](latent_basal)
            covars_pred[covar] = covar_pred

        if self.loss_ae == 'nb':
            dist_px = self.decoder(inputs=latent, px_r=self.px_r)
            return dict(
                dist_px=dist_px,
                drugs_pred=drugs_pred,
                covars_pred=covars_pred,
            )

        else:
            means, variances = self.decoder(inputs=latent)
            return dict(
                means=means,
                variances=variances,
                drugs_pred=drugs_pred,
                covars_pred=covars_pred,
            )

    def loss(self, tensors, inference_outputs, generative_outputs):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        x = tensors[_CE_CONSTANTS.X_KEY]
        # x = inference_outputs["x"]
        
        # Reconstruction loss & regularizations
        means = generative_outputs["means"]
        variances = generative_outputs["variances"]

        # log_px = dist_px.log_prob(x).sum(-1)
        # Compute reconstruction
        # reconstruction_loss = -log_px
        if self.loss_ae in ["gauss", "mse"]:
            # TODO: Check with Normal Distribution
            # variance = dist_px.scale ** 2
            # mean = dist_px.loc
            term1 = variances.log().div(2)
            term2 = (x - means).pow(2).div(variances.mul(2))

            reconstruction_loss = (term1 + term2).mean()
            # term1 = variance.log().div(2)
            # term2 = (x - mean).pow(2).div(variance.mul(2))
            # reconstruction_loss = (term1 + term2).mean()

        # elif self.loss_ae == 'mse':
        #     reconstruction_loss = (x - means).pow(2)

        # TODO: Add KL annealing if needed
        # kl_loss = 0.0
        # if self.variational:
        #     dist_qz = inference_outputs["dist_qz"]
        #     dist_pz = db.Normal(
        #         torch.zeros_like(dist_qz.loc), torch.ones_like(dist_qz.scale)
        #     )
        #     kl_loss = kl_divergence(dist_qz, dist_pz).sum(-1)
            # loss = -log_px + kl_z
        # else:
        

        return reconstruction_loss

    def adversarial_loss(self, tensors, inference_outputs, generative_outputs):
        """Computes adversarial classification losses and regularizations"""
        drugs_doses = tensors[_CE_CONSTANTS.PERTURBATIONS]

        latent_basal = inference_outputs["latent_basal"]
        covars_dict = inference_outputs["covars_dict"]

        drugs_pred = generative_outputs["drugs_pred"]
        covars_pred = generative_outputs["covars_pred"]

        adv_results = {}

        # Classification losses for different covariates
        for covar in self.covars_to_ncovars.keys():
            adv_results[f'adv_{covar}'] = self.adv_loss_covariates[covar](
                covars_pred[covar],
                covars_dict[covar].long().squeeze(-1),
            )
        
        # Classification loss for different drug combinations
        adv_results['adv_drugs'] = self.adv_loss_drugs(drugs_pred, drugs_doses.gt(0).float())
        adv_results['adv_loss'] = adv_results['adv_drugs'] + sum([adv_results[f'adv_{key}'] for key in self.covars_to_ncovars.keys()])

        # Penalty losses
        for covar in self.covars_to_ncovars.keys():
            adv_results[f'penalty_{covar}'] = (
                torch.autograd.grad(
                    covars_pred[covar].sum(), 
                    latent_basal, 
                    create_graph=True
                )[0].pow(2).mean()
            )

        adv_results['penalty_drugs'] = (
            torch.autograd.grad(
                drugs_pred.sum(),
                latent_basal,
                create_graph=True,
            )[0].pow(2).mean()
        )
        adv_results['penalty_adv'] = adv_results['penalty_drugs'] + sum([adv_results[f'penalty_{covar}'] for covar in self.covars_to_ncovars.keys()])

        return adv_results

    def r2_metric(self, tensors, inference_outputs, generative_outputs):
        pred_mean = generative_outputs['means'].detach().cpu().numpy() # batch_size, n_genes
        pred_var = generative_outputs['variances'].detach().cpu().numpy() # batch_size, n_genes

        x = tensors[_CE_CONSTANTS.X_KEY].detach().cpu().numpy() # batch_size, n_genes
        
        true_mean = x.mean(0)
        pred_mean = pred_mean.mean(0)

        true_var = x.var(0)
        pred_var = pred_var.mean(0)

        r2_mean = r2_score(true_mean, pred_mean)
        r2_var = r2_score(true_var, pred_var)

        return r2_mean, r2_var

    def disentanglement(self, tensors, inference_outputs, generative_outputs, linear=True):
        latent_basal = inference_outputs['latent_basal'].detach().cpu().numpy()
        drug_names = tensors['drug_name'].detach().cpu().numpy()
        
        classifier = LogisticRegression(solver="liblinear",
                                        multi_class="auto",
                                        max_iter=10000)

        
        pert_scores = cross_val_score(classifier, 
                                      StandardScaler().fit_transform(latent_basal), 
                                      drug_names.ravel(), 
                                      scoring=make_scorer(balanced_accuracy_score), 
                                      cv=5, 
                                      n_jobs=-1)
        
        return pert_scores.mean(), 0.0
    

    def get_expression(self, tensors, **inference_kwargs):
        """Computes gene expression means and std.

        Only implemented for the gaussian likelihood.

        Parameters
        ----------
        tensors : dict
            Considered inputs

        """
        _, generative_outputs = self.forward(
            tensors,
            compute_loss=False,
            inference_kwargs=inference_kwargs,
        )
        if self.loss_ae in ["gauss", 'mse']:
            mus = generative_outputs["means"]
            stds = generative_outputs["variances"]
            return mus, stds
        else:
            raise ValueError
