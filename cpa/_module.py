from sklearn.metrics import r2_score, make_scorer, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.distributions as db
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, FCLayers
from scvi.module import Classifier
from scvi import settings

from ._utils import _CE_CONSTANTS, DecoderGauss, DecoderNB, DrugNetwork, SimpleEncoder

import numpy as np


class CPAModule(BaseModuleClass):
    """
    CPA module using Gaussian/NegativeBinomial Likelihood

    Parameters
    ----------
        n_genes: int
        n_treatments: int
        covars_encoder: dict
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
                 covars_encoder: dict,
                 n_latent: int = 128,
                 loss_ae="gauss",
                 doser_type="logsigm",
                 output_activation: str = 'linear',
                 autoencoder_width=256,
                 autoencoder_depth=3,
                 adversary_width=64,
                 adversary_depth=2,
                 dosers_width: int = 128,
                 dosers_depth: int = 2,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 dropout_rate: float = 0.0,
                 variational: bool = False,
                 combinatorial: bool = False,
                 seed: int = 0,
                 ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

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
        self.combinatorial = combinatorial

        self.covars_encoder = covars_encoder

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
                dropout_rate=dropout_rate,
                activation_fn=nn.ReLU,
            )
        else:
            self.encoder = SimpleEncoder(
                n_input=n_genes,
                n_output=n_latent,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
                activation_fn=nn.ReLU,
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
                output_activation=output_activation,
                dropout_rate=dropout_rate,
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
                dropout_rate=dropout_rate,
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
                                        dropout_rate=dropout_rate,
                                        )

        self.drugs_classifier = Classifier(
            n_input=n_latent,
            n_labels=n_drugs,
            n_hidden=self.adversary_width,
            n_layers=self.adversary_depth,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            activation_fn=nn.ReLU,
            logits=True,
        )

        # 2. Covariates Embedding
        self.covars_embedding = nn.ModuleDict(
            {
                key: torch.nn.Embedding(len(unique_covars), n_latent)
                for key, unique_covars in self.covars_encoder.items()
            }
        )

        self.covars_classifiers = nn.ModuleDict(
            {
                key: Classifier(n_input=n_latent,
                                n_labels=len(unique_covars),
                                n_hidden=self.adversary_width,
                                n_layers=self.adversary_depth,
                                use_batch_norm=use_batch_norm,
                                use_layer_norm=use_layer_norm,
                                dropout_rate=dropout_rate,
                                logits=True)
                if len(unique_covars) > 1 else None

                for key, unique_covars in self.covars_encoder.items()
            }
        )

        self.ae_loss_fn = nn.GaussianNLLLoss()
        self.adv_loss_covariates = nn.CrossEntropyLoss()

        if self.combinatorial:
            self.adv_loss_drugs = nn.BCEWithLogitsLoss()
        else:
            self.adv_loss_drugs = nn.CrossEntropyLoss()

    def _get_inference_input(self, tensors):
        x = tensors[_CE_CONSTANTS.X_KEY]  # batch_size, n_genes
        if self.combinatorial:
            drugs_doses = tensors['drugs_doses']
        else:
            drugs = tensors[f"drug_name"]
            doses = tensors[f"dose_value"]

        covars_dict = dict()
        for covar, unique_covars in self.covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  # (batch_size,)
            covars_dict[covar] = encoded_covars

        input_dict = dict(
            genes=x,
            drugs=drugs_doses if self.combinatorial else drugs,
            doses=None if self.combinatorial else doses,
            covars_dict=covars_dict,
        )
        return input_dict

    @auto_move_data
    def inference(
            self,
            genes,
            drugs,
            doses,
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
        for covar, _ in self.covars_encoder.items():
            latent_covar_i = self.covars_embedding[covar](covars_dict[covar].long())
            latent_covar_i = latent_covar_i.view(batch_size, self.n_latent).unsqueeze(0)  # 1, batch_size, n_latent
            latent_covariates.append(latent_covar_i)

        latent_covariates = torch.cat(latent_covariates, 0).sum(0)  # Summing all covariates representations
        latent_treatment = self.drug_network(drugs, doses)
        latent = latent_basal + latent_covariates + latent_treatment

        return dict(
            latent=latent,
            latent_basal=latent_basal,
            dist_qz=dist_qzbasal,
            library=library,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        input_dict = {}

        latent = inference_outputs["latent"]
        latent_basal = inference_outputs['latent_basal']
        if self.loss_ae == 'nb':
            library = inference_outputs["library"]
            input_dict["library"] = library

        covars_dict = dict()
        for covar, _ in self.covars_encoder.items():
            val = tensors[covar].view(-1, )
            covars_dict[covar] = val

        input_dict['latent'] = latent
        input_dict['latent_basal'] = latent_basal
        return input_dict

    @auto_move_data
    def generative(
            self,
            latent,
            latent_basal,
    ):
        if self.loss_ae == 'nb':
            dist_px = self.decoder(inputs=latent, px_r=self.px_r)
            return dict(
                dist_px=dist_px,
            )

        else:
            outputs = self.decoder(inputs=latent)
            return dict(
                means=outputs.loc,
                variances=outputs.variance,
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
            reconstruction_loss = self.ae_loss_fn(x, means, variances)
            # variance = dist_px.scale ** 2
            # mean = dist_px.loc
            # term1 = variances.log().div(2)
            # term2 = (x - means).pow(2).div(variances.mul(2))
            #
            # reconstruction_loss = (term1 + term2).mean()
            # term1 = variance.log().div(2)
            # term2 = (x - mean).pow(2).div(variance.mul(2))
            # reconstruction_loss = (term1 + term2).mean()
        else:
            raise Exception('Invalid Loss function for CPA')
        # elif self.loss_ae == 'mse':
        #     reconstruction_loss = (x - means).pow(2)

        # TODO: Add KL annealing if needed
        kl_loss = 0.0
        if self.variational:
            dist_qz = inference_outputs["dist_qz"]
            dist_pz = db.Normal(
                torch.zeros_like(dist_qz.loc), torch.ones_like(dist_qz.scale)
            )
            kl_loss = kl_divergence(dist_qz, dist_pz).sum(-1)

        return reconstruction_loss, kl_loss

    def adversarial_loss(self, tensors, latent_basal):
        """Computes adversarial classification losses and regularizations"""
        if self.combinatorial:
            drugs_doses = tensors['drugs_doses']
            batch_size = drugs_doses.shape[0]
        else:
            drugs = tensors[f"drug_name"].view(-1, )
            batch_size = drugs.shape[0]

        covars_dict = dict()
        for covar, unique_covars in self.covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  # (batch_size,)
            covars_dict[covar] = encoded_covars

        drugs_pred = self.drugs_classifier(latent_basal)
        covars_pred = {}
        for covar in self.covars_encoder.keys():
            if self.covars_classifiers[covar] is not None:
                covar_pred = self.covars_classifiers[covar](latent_basal)
                covars_pred[covar] = covar_pred
            else:
                covars_pred[covar] = None

        adv_results = {}
        # Classification losses for different covariates
        for covar in self.covars_encoder.keys():
            adv_results[f'adv_{covar}'] = self.adv_loss_covariates(
                covars_pred[covar],
                covars_dict[covar].long(),
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)
            adv_results[f'acc_{covar}'] = torch.sum(
                covars_pred[covar].argmax(1) == covars_dict[covar].long().squeeze(-1)) / batch_size \
                if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        # Classification loss for different drug combinations
        adv_results['adv_drugs'] = self.adv_loss_drugs(drugs_pred, drugs_doses.gt(
            0).float() if self.combinatorial else drugs.long())
        adv_results['adv_loss'] = adv_results['adv_drugs'] + sum(
            [adv_results[f'adv_{key}'] for key in self.covars_encoder.keys()])

        # Penalty losses
        for covar in self.covars_encoder.keys():
            adv_results[f'penalty_{covar}'] = (
                torch.autograd.grad(
                    covars_pred[covar].sum(),
                    latent_basal,
                    create_graph=True
                )[0].pow(2).mean()
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        adv_results['penalty_drugs'] = (
            torch.autograd.grad(
                drugs_pred.sum(),
                latent_basal,
                create_graph=True,
            )[0].pow(2).mean()
        )
        adv_results['penalty_adv'] = adv_results['penalty_drugs'] + sum(
            [adv_results[f'penalty_{covar}'] for covar in self.covars_encoder.keys()])

        return adv_results

    def r2_metric(self, tensors, inference_outputs, generative_outputs):
        pred_mean = torch.nan_to_num(generative_outputs['means'], nan=1e2, neginf=-1e3,
                                     posinf=1e3).detach().cpu().numpy()  # batch_size, n_genes
        pred_var = torch.nan_to_num(generative_outputs['variances'], nan=1e2, neginf=-1e3,
                                    posinf=1e3).detach().cpu().numpy()  # batch_size, n_genes

        x = tensors[_CE_CONSTANTS.X_KEY].detach().cpu().numpy()  # batch_size, n_genes

        true_mean = x.mean(0)
        pred_mean = pred_mean.mean(0)

        true_var = x.var(0)
        pred_var = pred_var.mean(0)

        r2_mean = r2_score(true_mean, pred_mean)
        r2_var = r2_score(true_var, pred_var)

        return r2_mean, r2_var

    def disentanglement(self, tensors, inference_outputs, generative_outputs, linear=True):
        latent_basal = inference_outputs['latent_basal'].detach().cpu().numpy()
        latent = inference_outputs['latent'].detach().cpu().numpy()
        if self.combinatorial:
            drugs_doses = tensors['drugs_doses']
            drug_names = drugs_doses.argmax(dim=1).float().detach().cpu().numpy()
        else:
            drug_names = tensors['drug_name'].detach().cpu().numpy()

        classifier = LogisticRegression(solver="liblinear",
                                        multi_class="auto",
                                        max_iter=10000)

        pert_basal_scores = cross_val_score(classifier,
                                            StandardScaler().fit_transform(latent_basal),
                                            drug_names.ravel(),
                                            scoring=make_scorer(balanced_accuracy_score),
                                            cv=min(5, len(np.unique(drug_names.ravel()))),
                                            n_jobs=-1).mean()

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                target_covars = tensors[f'{covar}'].detach().cpu().numpy()
                classifier = LogisticRegression(solver="liblinear",
                                                multi_class="auto",
                                                max_iter=10000)

                pert_basal_scores += cross_val_score(classifier,
                                                     StandardScaler().fit_transform(latent_basal),
                                                     target_covars.ravel(),
                                                     scoring=make_scorer(balanced_accuracy_score),
                                                     cv=min(5, len(np.unique(target_covars.ravel()))),
                                                     n_jobs=-1).mean()

        classifier = LogisticRegression(solver="liblinear",
                                        multi_class="auto",
                                        max_iter=10000)

        pert_scores = cross_val_score(classifier,
                                      StandardScaler().fit_transform(latent),
                                      drug_names.ravel(),
                                      scoring=make_scorer(balanced_accuracy_score),
                                      cv=5,
                                      n_jobs=-1).mean()

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                target_covars = tensors[f'{covar}'].detach().cpu().numpy()
                classifier = LogisticRegression(solver="liblinear",
                                                multi_class="auto",
                                                max_iter=10000)

                pert_scores += cross_val_score(classifier,
                                               StandardScaler().fit_transform(latent),
                                               target_covars.ravel(),
                                               scoring=make_scorer(balanced_accuracy_score),
                                               cv=min(5, len(np.unique(target_covars.ravel()))),
                                               n_jobs=-1).mean()

        return pert_basal_scores, pert_scores

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
            mus = torch.nan_to_num(generative_outputs['means'], nan=1e2, neginf=-1e3,
                                   posinf=1e3)  # batch_size, n_genes
            stds = torch.nan_to_num(generative_outputs['variances'], nan=1e2, neginf=-1e3,
                                    posinf=1e3)  # batch_size, n_genes
            return mus, stds
        else:
            raise ValueError

    def get_drug_embeddings(self, tensors, **inference_kwargs):
        inputs = self._get_inference_input(tensors)
        drugs = inputs['drugs']

        if self.combinatorial:
            doses = None
        else:
            doses = inputs['doses']

        return self.drug_network(drugs, doses)
