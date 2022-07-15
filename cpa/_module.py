from sklearn.metrics import r2_score

import torch
import torch.distributions as db
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, FCLayers
from scvi.module import Classifier
from scvi import settings

from ._metrics import entropy_batch_mixing, knn_purity
from ._utils import DecoderNormal, DrugNetwork, VanillaEncoder, CPA_REGISTRY_KEYS, DecoderNB

import numpy as np
from copy import deepcopy


class CPAModule(BaseModuleClass):
    """
    CPA module using Gaussian/NegativeBinomial Likelihood

    Parameters
    ----------
        n_genes: int
        n_treatments: int
        cat_covars_encoder: dict
            Dictionary of covariates with keys as each covariate name and values as
                number of unique values of the corresponding covariate
        n_latent: int
            Latent Dimension
        loss_ae: str
            Autoencoder loss (either "gauss", "mse" or "nb")
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
                 cat_covars_encoder: dict,
                 cont_covars: list = [],
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
                 seed: int = 0,
                 ):
        super().__init__()

        loss_ae = loss_ae.lower()
        assert loss_ae in ['mse', 'gauss', 'nb']

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

        self.cat_covars_encoder = cat_covars_encoder
        self.cont_covars = cont_covars

        self.control_treatment_idx = None

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
            self.encoder = VanillaEncoder(
                n_input=n_genes,
                n_output=n_latent,
                n_cat_list=[],
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
                dropout_rate=dropout_rate,
                activation_fn=nn.ReLU,
            )

            # Decoder components
            self.px_r = torch.nn.Parameter(torch.randn(n_genes))

        # Decoder components
        if loss_ae in ["gauss", 'mse']:
            self.decoder = DecoderNormal(
                n_input=n_latent,
                n_output=n_genes,
                n_cat_list=[],
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                output_activation=output_activation,
                dropout_rate=dropout_rate,
            )

        elif loss_ae == 'nb':
            self.decoder = DecoderNB(
                n_input=n_latent,
                n_output=n_genes,
                n_hidden=autoencoder_width,
                n_layers=autoencoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                output_activation=output_activation,
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
        self.cat_covars_embeddings = nn.ModuleDict(
            {
                key: torch.nn.Embedding(len(unique_covars), n_latent)
                for key, unique_covars in self.cat_covars_encoder.items()
            }
        )

        self.cat_covars_classifiers = nn.ModuleDict(
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

                for key, unique_covars in self.cat_covars_encoder.items()
            }
        )

        self.cont_covars_embeddings = nn.ModuleDict(
            {
                key: torch.nn.Linear(1, n_latent, bias=False) for key in self.cont_covars
            }
        )

        self.cont_covars_regressors = nn.ModuleDict(
            {
                key: FCLayers(n_in=n_latent,
                              n_out=1,
                              n_hidden=self.adversary_width,
                              n_layers=self.adversary_depth,
                              use_batch_norm=use_batch_norm,
                              use_layer_norm=use_layer_norm,
                              dropout_rate=dropout_rate,
                              activation_fn=nn.ReLU,
                              )
                for key in self.cont_covars
            }
        )

        self.ae_loss_fn = nn.GaussianNLLLoss()
        self.adv_loss_cat_covariates = nn.CrossEntropyLoss()
        self.adv_loss_cont_covariates = nn.MSELoss()

        self.adv_loss_drugs = nn.BCEWithLogitsLoss()

    def _get_inference_input(self, tensors):
        x = tensors[CPA_REGISTRY_KEYS.X_KEY]  # batch_size, n_genes
        drugs_doses = tensors['drugs_doses']

        cat_covars_dict = dict()
        for covar, unique_covars in self.cat_covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  # (batch_size,)
            cat_covars_dict[covar] = encoded_covars

        cont_covars_dict = dict()
        for covar in self.cont_covars:
            encoded_covars = tensors[covar].view(-1, )  # (batch_size,)
            cont_covars_dict[covar] = encoded_covars

        input_dict = dict(
            genes=x,
            drugs=drugs_doses,
            doses=None,
            cat_covars_dict=cat_covars_dict,
            cont_covars_dict=cont_covars_dict,
        )
        return input_dict

    @auto_move_data
    def inference(
            self,
            genes,
            drugs,
            doses,
            cat_covars_dict,
            cont_covars_dict,
    ):
        # x_ = torch.log1p(x)
        batch_size = genes.shape[0]
        x_ = genes
        if self.loss_ae == 'nb':
            x_ = torch.log1p(x_)

        if self.variational:
            z_means, z_vars, latent_basal = self.encoder(x_)
            basal_distribution = db.Normal(z_means, z_vars.sqrt())
        else:
            basal_distribution = None
            latent_basal = self.encoder(x_)

        if self.loss_ae == 'nb':
            library = self.l_encoder(x_)
        else:
            library = None

        latent_treatment = self.drug_network(drugs, doses)

        latent_cat_covariates = []
        for covar, _ in self.cat_covars_encoder.items():
            latent_covar_i = self.cat_covars_embeddings[covar](cat_covars_dict[covar].long())
            latent_covar_i = latent_covar_i.view(batch_size, self.n_latent).unsqueeze(0)  # 1, batch_size, n_latent
            latent_cat_covariates.append(latent_covar_i)

        latent_cont_covariates = []
        for covar, _ in self.cont_covars:
            latent_covar_i = self.cont_covars_embeddings[covar](cont_covars_dict[covar])
            latent_covar_i = latent_covar_i.view(batch_size, self.n_latent).unsqueeze(0)  # 1, batch_size, n_latent
            latent_cont_covariates.append(latent_covar_i)

        latent = latent_basal + latent_treatment

        if len(latent_cat_covariates) > 0:
            latent_cat_covariates = torch.cat(latent_cat_covariates, 0).sum(
                0)  # Summing all categorical covariates representations
            latent += latent_cat_covariates
        else:
            latent_cat_covariates = None

        if len(latent_cont_covariates) > 0:
            latent_cont_covariates = torch.cat(latent_cont_covariates, 0).sum(
                0)  # Summing all continuous covariates representations
            latent += latent_cont_covariates
        else:
            latent_cont_covariates = None

        return dict(
            latent=latent,
            latent_basal=latent_basal,
            latent_treatment=latent_treatment,
            latent_cont_covariates=latent_cont_covariates,
            latent_cat_covariates=latent_cat_covariates,
            library=library,
            basal_distribution=basal_distribution,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        input_dict = {}

        latent = inference_outputs["latent"]
        latent_basal = inference_outputs['latent_basal']

        covars_dict = dict()
        for covar, _ in self.cat_covars_encoder.items():
            val = tensors[covar].view(-1, )
            covars_dict[covar] = val

        input_dict['latent'] = latent
        input_dict['latent_basal'] = latent_basal
        input_dict['library'] = inference_outputs['library']
        return input_dict

    @auto_move_data
    def generative(
            self,
            latent,
            latent_basal,
            library=None,
    ):
        if self.loss_ae in ['mse', 'gauss']:
            outputs = self.decoder(inputs=latent)
            return dict(
                means=outputs.loc,
                variances=outputs.variance,
                distribution=outputs,
                samples=outputs.sample_n(n=1).squeeze(0),
            )
        elif self.loss_ae == 'nb':
            outputs = self.decoder(inputs=latent, library=library, px_r=self.px_r)
            return dict(
                distribution=outputs,
                samples=outputs.sample().squeeze(0),
            )

    def loss(self, tensors, inference_outputs, generative_outputs):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        x = tensors[CPA_REGISTRY_KEYS.X_KEY]

        if self.loss_ae in ["gauss", "mse"]:
            means = generative_outputs["means"]
            variances = generative_outputs["variances"]
            reconstruction_loss = self.ae_loss_fn(x, means, variances)
        elif self.loss_ae == 'nb':
            dist_px = generative_outputs['distribution']
            log_px = dist_px.log_prob(x).mean(-1)
            reconstruction_loss = -log_px
            reconstruction_loss = reconstruction_loss.mean()
        else:
            raise Exception('Invalid Loss function for CPA')

        # TODO: Add KL annealing if needed
        kl_loss = 0.0
        if self.variational:
            basal_distribution = inference_outputs["basal_distribution"]
            dist_pz = db.Normal(
                torch.zeros_like(basal_distribution.loc), torch.ones_like(basal_distribution.scale)
            )
            kl_loss = kl_divergence(basal_distribution, dist_pz).sum(-1)

        return reconstruction_loss, kl_loss

    def cycle_regularization(self, tensors, inference_outputs, generative_outputs):
        x = tensors[CPA_REGISTRY_KEYS.X_KEY]

        batch_size = x.shape[0]

        if self.control_treatment_idx is None:
            control_mask = tensors['control'].bool().view(-1, )
            self.control_treatment_idx = tensors['drugs_doses'][control_mask][0, :].view(1, -1)

        control_latent = self.drug_network(self.control_treatment_idx.repeat(batch_size, 1))
        latent_control_basal = inference_outputs['latent_basal'] + control_latent

        if inference_outputs['latent_cat_covariates'] is not None:
            latent_control_basal += inference_outputs['latent_cat_covariates']

        if inference_outputs['latent_cont_covariates'] is not None:
            latent_control_basal += inference_outputs['latent_cont_covariates']

        control_outputs = self.generative(latent_control_basal, None, library=inference_outputs['library'])

        pred_control_x = torch.nan_to_num(control_outputs['samples'], neginf=0, nan=0, posinf=100)

        new_tensors = deepcopy(tensors)
        new_tensors[CPA_REGISTRY_KEYS.X_KEY] = pred_control_x

        pred_x = self.generative(
            **self._get_generative_input(new_tensors, self.inference(**self._get_inference_input(new_tensors))))

        if 'deg_mask' in tensors:
            deg_mask = tensors['deg_mask']

            x = x * deg_mask
            if self.loss_ae in ['gauss', 'mse']:
                pred_x['means'] *= deg_mask
                pred_x['variances'] *= deg_mask

        if self.loss_ae in ['gauss', 'mse']:
            reconstruction_loss = self.ae_loss_fn(x, pred_x['means'], pred_x['variances'])

        elif self.loss_ae == 'nb':
            dist_px = generative_outputs['distribution']
            log_px = dist_px.log_prob(x).mean(-1)
            reconstruction_loss = -log_px
            reconstruction_loss = reconstruction_loss.mean()
        else:
            raise ValueError('Incorrect loss_ae has been passed! Must be either `gauss`, `mse`, or `nb`')

        return reconstruction_loss

    def adversarial_loss(self, tensors, latent_basal):
        """Computes adversarial classification losses and regularizations"""
        drugs_doses = tensors['drugs_doses']
        batch_size = drugs_doses.shape[0]

        drugs_pred = self.drugs_classifier(latent_basal)
        cat_covars_pred = {}
        for covar in self.cat_covars_encoder.keys():
            if self.cat_covars_classifiers[covar] is not None:
                covar_pred = self.cat_covars_classifiers[covar](latent_basal)
                cat_covars_pred[covar] = covar_pred
            else:
                cat_covars_pred[covar] = None

        cont_covars_pred = {}
        for covar in self.cont_covars:
            if self.cont_covars_regressors[covar] is not None:
                covar_pred = self.cont_covars_regressors[covar](latent_basal)
                cont_covars_pred[covar] = covar_pred
            else:
                cont_covars_pred[covar] = None

        adv_results = {}

        # Classification losses for different categorical covariates
        for covar in self.cat_covars_encoder.keys():
            adv_results[f'adv_{covar}'] = self.adv_loss_cat_covariates(
                cat_covars_pred[covar],
                tensors[covar].view(-1, ).long(),
            ) if cat_covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)
            adv_results[f'acc_{covar}'] = torch.sum(
                cat_covars_pred[covar].argmax(1) == tensors[covar].long().view(-1, )) / batch_size \
                if cat_covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        # Regression losses for different continuous covariates
        for covar in self.cont_covars:
            adv_results[f'adv_{covar}'] = self.adv_loss_cont_covariates(
                cont_covars_pred[covar],
                tensors[covar].view(-1, ),
            ) if cont_covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        # Classification loss for different drug combinations
        adv_results['adv_drugs'] = self.adv_loss_drugs(drugs_pred, drugs_doses.gt(
            0).float())
        adv_results['adv_loss'] = \
            adv_results['adv_drugs'] + \
            sum([adv_results[f'adv_{key}'] for key in self.cat_covars_encoder.keys()]) + \
            sum([adv_results[f'adv_{key}'] for key in self.cont_covars])

        # Penalty losses
        for covar in self.cat_covars_encoder.keys():
            adv_results[f'penalty_{covar}'] = (
                torch.autograd.grad(
                    cat_covars_pred[covar].sum(),
                    latent_basal,
                    create_graph=True
                )[0].pow(2).mean()
            ) if cat_covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        for covar in self.cont_covars:
            adv_results[f'penalty_{covar}'] = (
                torch.autograd.grad(
                    cont_covars_pred[covar].sum(),
                    latent_basal,
                    create_graph=True
                )[0].pow(2).mean()
            ) if cont_covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        adv_results['penalty_drugs'] = (
            torch.autograd.grad(
                drugs_pred.sum(),
                latent_basal,
                create_graph=True,
            )[0].pow(2).mean()
        )

        adv_results['penalty_adv'] = \
            adv_results['penalty_drugs'] + \
            sum([adv_results[f'penalty_{covar}'] for covar in self.cat_covars_encoder.keys()]) + \
            sum([adv_results[f'penalty_{covar}'] for covar in self.cont_covars])

        return adv_results

    def r2_metric(self, tensors, inference_outputs, generative_outputs, method: str = 'lfc'):
        method = method.lower()
        assert method.lower() in ['lfc', 'cycle', 'abs']

        x = tensors[CPA_REGISTRY_KEYS.X_KEY].detach().cpu().numpy()  # batch_size, n_genes

        batch_size = x.shape[0]

        if self.control_treatment_idx is None:
            control_mask = tensors['control'].bool().view(-1, )
            self.control_treatment_idx = tensors['drugs_doses'][control_mask][0, :].view(1, -1)

        control_latent = self.drug_network(self.control_treatment_idx.repeat(batch_size, 1))

        latent_control_basal = inference_outputs['latent_basal'] + control_latent

        if inference_outputs['latent_cat_covariates'] is not None:
            latent_control_basal += inference_outputs['latent_cat_covariates']

        if inference_outputs['latent_cont_covariates'] is not None:
            latent_control_basal += inference_outputs['latent_cont_covariates']

        control_outputs = self.generative(latent_control_basal, inference_outputs['latent_basal'],
                                          library=inference_outputs['library'])

        drugs_doses = tensors['drugs_doses']  # (batch_size, n_drugs)
        indices = (drugs_doses * torch.arange(drugs_doses.shape[1]).view(1, -1).repeat(batch_size, 1).to(
            drugs_doses.device)).sum(dim=1)

        for i, covar in enumerate(self.cat_covars_encoder.keys()):
            indices += tensors[covar].view(-1, )  # (batch_size,)

        r2_mean = 0.0
        r2_var = 0.0

        unique_indices = indices.unique()
        n_unique_indices = len(unique_indices)

        if method == 'lfc':
            if self.loss_ae in ['gauss', 'mse']:
                pred_control_mean = torch.nan_to_num(control_outputs['means'], neginf=0, nan=0,
                                                     posinf=100).detach().cpu().numpy()
                pred_control_var = torch.nan_to_num(control_outputs['variances'], neginf=0, nan=0,
                                                    posinf=100).detach().cpu().numpy()

                pred_x_mean = torch.nan_to_num(generative_outputs['means'], neginf=0, nan=0,
                                               posinf=100).detach().cpu().numpy()
                pred_x_var = torch.nan_to_num(generative_outputs['variances'], neginf=0, nan=0,
                                              posinf=100).detach().cpu().numpy()

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x_mean *= deg_mask
                    pred_x_var *= deg_mask
                    pred_control_mean *= deg_mask
                    pred_control_var *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_mean_index = pred_x_mean[index_mask]
                    pred_x_var_index = pred_x_var[index_mask]

                    pred_control_mean_index = pred_control_mean[index_mask]
                    pred_control_var_index = pred_control_var[index_mask]

                    true_mean_index = x_index.mean(0) - np.nan_to_num(pred_control_mean_index.mean(0), neginf=0.,
                                                                      posinf=100., nan=0.0)
                    true_var_index = x_index.var(0) - np.nan_to_num(pred_control_var_index.mean(0), neginf=0.,
                                                                    posinf=100., nan=0.0)

                    pred_mean_index = np.nan_to_num(pred_x_mean_index.mean(0) - pred_control_mean_index.mean(0),
                                                    neginf=0., posinf=100., nan=0.0)
                    pred_var_index = np.nan_to_num(pred_x_var_index.mean(0) - pred_control_var_index.mean(0), neginf=0.,
                                                   posinf=100., nan=0.0)

                    r2_mean += r2_score(true_mean_index, pred_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_var_index) / n_unique_indices

            elif self.loss_ae == 'nb':
                pred_control = torch.nan_to_num(control_outputs['samples'], neginf=0, nan=0,
                                                posinf=100).detach().cpu().numpy()

                pred_x = torch.nan_to_num(generative_outputs['samples'], neginf=0, nan=0,
                                          posinf=100).detach().cpu().numpy()

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x *= deg_mask
                    pred_control *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_index = pred_x[index_mask]

                    pred_x_mean_index = np.nan_to_num(pred_x_index.mean(0), neginf=0., posinf=100., nan=0.0)
                    pred_x_var_index = np.nan_to_num(pred_x_index.var(0), neginf=0., posinf=100., nan=0.0)

                    pred_control_mean_index = np.nan_to_num(pred_control[index_mask].mean(0),
                                                            neginf=0., posinf=100.,
                                                            nan=0.0)
                    pred_control_var_index = np.nan_to_num(pred_control[index_mask].var(0),
                                                           neginf=0., posinf=100.,
                                                           nan=0.0)

                    true_mean_index = np.nan_to_num(np.log1p(x_index), neginf=0.) - np.nan_to_num(
                        np.log1p(pred_control_mean_index), neginf=0.)
                    true_var_index = np.nan_to_num(np.log1p(x_index), neginf=0.) - np.nan_to_num(
                        np.log1p(pred_control_var_index), neginf=0.)

                    pred_mean_index = np.nan_to_num(np.log1p(pred_x_mean_index), neginf=0.) - np.nan_to_num(
                        np.log1p(pred_control_mean_index), neginf=0.)
                    pred_var_index = np.nan_to_num(np.log1p(pred_x_var_index), neginf=0.) - np.nan_to_num(
                        np.log1p(pred_control_var_index), neginf=0.)

                    r2_mean += r2_score(true_mean_index, pred_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_var_index) / n_unique_indices

        elif method == 'cycle':
            pred_control_x = torch.nan_to_num(control_outputs['samples'], neginf=0, nan=0, posinf=100)

            new_tensors = deepcopy(tensors)
            new_tensors[CPA_REGISTRY_KEYS.X_KEY] = pred_control_x

            pred_x = self.generative(
                **self._get_generative_input(new_tensors, self.inference(**self._get_inference_input(new_tensors))))

            if self.loss_ae in ['gauss', 'mse']:
                pred_x_mean = torch.nan_to_num(pred_x['means'], neginf=0., posinf=100., nan=0.0).detach().cpu().numpy()
                pred_x_var = torch.nan_to_num(pred_x['variances'], neginf=0., posinf=100.,
                                              nan=0.0).detach().cpu().numpy()

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x_mean *= deg_mask
                    pred_x_var *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_mean_index = pred_x_mean[index_mask]
                    pred_x_var_index = pred_x_var[index_mask]

                    true_mean_index = x_index.mean(0)
                    true_var_index = x_index.var(0)

                    pred_mean_index = np.nan_to_num(pred_x_mean_index.mean(0), neginf=0.0, posinf=100, nan=0.0)
                    pred_var_index = np.nan_to_num(pred_x_var_index.mean(0), neginf=0.0, posinf=100, nan=0.0)

                    r2_mean += r2_score(true_mean_index, pred_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_var_index) / n_unique_indices
            elif self.loss_ae == 'nb':
                x = np.log1p(x)
                pred_control_x = torch.nan_to_num(control_outputs['samples'], neginf=0, nan=0, posinf=100)

                new_tensors = deepcopy(tensors)
                new_tensors[CPA_REGISTRY_KEYS.X_KEY] = pred_control_x

                pred_x = self.generative(
                    **self._get_generative_input(new_tensors, self.inference(**self._get_inference_input(new_tensors))))

                pred_x = np.log1p(
                    torch.nan_to_num(pred_x['samples'], neginf=0, posinf=100, nan=0).detach().cpu().numpy())

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_index = pred_x[index_mask]

                    pred_x_mean_index = np.nan_to_num(pred_x_index.mean(0), neginf=0., posinf=100., nan=0.0)
                    pred_x_var_index = np.nan_to_num(pred_x_index.var(0), neginf=0., posinf=100., nan=0.0)

                    true_mean_index = x_index.mean(0)
                    true_var_index = x_index.var(0)

                    r2_mean += r2_score(true_mean_index, pred_x_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_x_var_index) / n_unique_indices

        elif method == 'abs':
            if self.loss_ae in ['gauss', 'mse']:
                pred_x_mean = torch.nan_to_num(generative_outputs['means'], neginf=0, nan=0,
                                               posinf=100).detach().cpu().numpy()
                pred_x_var = torch.nan_to_num(generative_outputs['variances'], neginf=0, nan=0,
                                              posinf=100).detach().cpu().numpy()

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x_mean *= deg_mask
                    pred_x_var *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_mean_index = pred_x_mean[index_mask]
                    pred_x_var_index = pred_x_var[index_mask]

                    true_mean_index = x_index.mean(0)
                    true_var_index = x_index.var(0)

                    pred_mean_index = pred_x_mean_index.mean(0)
                    pred_var_index = pred_x_var_index.mean(0)

                    r2_mean += r2_score(true_mean_index, pred_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_var_index) / n_unique_indices
            elif self.loss_ae == 'nb':
                x = np.log1p(x)
                pred_x = torch.nan_to_num(generative_outputs['samples'], neginf=0, nan=0,
                                          posinf=100).detach().cpu().numpy()

                pred_x = np.log1p(pred_x)

                if 'deg_mask' in tensors:
                    deg_mask = tensors['deg_mask'].detach().cpu().numpy()

                    x *= deg_mask
                    pred_x *= deg_mask

                for index in unique_indices:
                    index_mask = (indices == index).detach().cpu().numpy()

                    x_index = x[index_mask]
                    pred_x_index = pred_x[index_mask]

                    pred_x_mean_index = pred_x_index.mean(0)
                    pred_x_var_index = pred_x_index.var(0)

                    true_mean_index = x_index.mean(0)
                    true_var_index = x_index.var(0)

                    r2_mean += r2_score(true_mean_index, pred_x_mean_index) / n_unique_indices
                    r2_var += r2_score(true_var_index, pred_x_var_index) / n_unique_indices

        return r2_mean, r2_var

    def disentanglement(self, tensors, inference_outputs, generative_outputs, linear=True):
        latent_basal = inference_outputs['latent_basal'].detach().cpu().numpy()
        latent = inference_outputs['latent'].detach().cpu().numpy()
        drugs_doses = tensors['drugs_doses']
        drug_names = drugs_doses.argmax(dim=1).float().detach().cpu().numpy()

        knn_basal = knn_purity(latent_basal, drug_names.ravel(), n_neighbors=min(drug_names.shape[0] - 1, 30))

        for covar, unique_covars in self.cat_covars_encoder.items():
            if len(unique_covars) > 1:
                target_covars = tensors[f'{covar}'].detach().cpu().numpy()
                knn_basal += knn_purity(latent_basal, target_covars.ravel(),
                                        n_neighbors=min(target_covars.shape[0] - 1, 30))

        knn_after = knn_purity(latent, drug_names.ravel(), n_neighbors=min(drug_names.shape[0] - 1, 30))

        for covar, unique_covars in self.cat_covars_encoder.items():
            if len(unique_covars) > 1:
                target_covars = tensors[f'{covar}'].detach().cpu().numpy()
                knn_after += knn_purity(latent, target_covars.ravel(), n_neighbors=min(target_covars.shape[0] - 1, 30))

        return knn_basal, knn_after

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
            mus = torch.nan_to_num(generative_outputs['means'], nan=0, neginf=0,
                                   posinf=100)  # batch_size, n_genes
            stds = torch.nan_to_num(generative_outputs['variances'], nan=0, neginf=0,
                                    posinf=100)  # batch_size, n_genes
            return mus, stds
        elif self.loss_ae == 'nb':
            pred_x = torch.nan_to_num(generative_outputs['samples'], neginf=0, nan=0,
                                      posinf=100).detach().cpu().numpy()
            return pred_x
        else:
            raise ValueError

    def get_drug_embeddings(self, tensors, **inference_kwargs):
        inputs = self._get_inference_input(tensors)
        drugs = inputs['drugs']
        doses = None

        return self.drug_network(drugs, doses)
