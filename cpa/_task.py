from typing import Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from scvi._compat import Literal
from scvi.module.base import BaseModuleClass
from scvi.nn import FCLayers
from scvi.train import TrainingPlan

import numpy as np

class CPATrainingPlan(TrainingPlan):
    def __init__(
        self,
        module: BaseModuleClass,
        covars_to_ncovars: dict,
        autoencoder_lr=1e-3,
        weight_decay=1e-6,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "reconstruction_loss_validation",
        lr_min: float = 0,
        adversary_steps: int = 3,
        reg_adversary: int = 5,
        penalty_adversary: int = 3,
        dosers_lr=1e-3,
        dosers_wd=1e-7,
        adversary_lr=3e-4,
        adversary_wd=1e-2,
        autoencoder_wd=1e-6,
        step_size_lr: int = 45,
    ):
        """Training plan for the CPA model"""
        super().__init__(
            module=module,
            lr=autoencoder_lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
        )

        self.covars_to_ncovars = covars_to_ncovars

        # adversarial_models_kwargs = dict(
        #     n_hidden=adversary_width,
        #     n_layers=adversary_depth,
        # )

        self.autoencoder_wd = autoencoder_wd
        self.autoencoder_lr = autoencoder_lr
        
        self.adversary_lr = adversary_lr
        self.adversary_wd = adversary_wd
        self.adversary_steps = adversary_steps
        self.reg_adversary = reg_adversary
        self.penalty_adversary = penalty_adversary
        
        self.dosers_lr = dosers_lr
        self.dosers_wd = dosers_wd

        self.step_size_lr = step_size_lr

        self.automatic_optimization = False
        self.iter_count = 0

        self.epoch_history = {
            'mode': [], 
            'epoch': [],
            'recon_loss': [], 
            'adv_loss': [], 
            'penalty_adv': [], 
            'adv_drugs': [], 
            'penalty_drugs': []
        }

        for covar in self.covars_to_ncovars.keys():
            self.epoch_history[f'adv_{covar}'] = []
            self.epoch_history[f'penalty_{covar}'] = []

        # Adversarial modules and hparams
        # self.covariates_adv_nn = nn.ModuleDict(
        #     {
        #         key: FCLayers(
        #             n_in=module.n_latent, n_out=n_cats, **adversarial_models_kwargs
        #         )
        #         for key, n_cats in module.cat_to_ncats.items()
        #     }
        # )
        # self.treatments_adv_nn = FCLayers(
        #     n_in=module.n_latent, n_out=module.n_treatments, **adversarial_models_kwargs
        # )
        # self.adv_loss_covariates = nn.CrossEntropyLoss()
        # self.adv_loss_treatments = nn.BCEWithLogitsLoss()
        

    # def _adversarial_classifications(self, z_basal):
    #     """Computes adversarial classifier predictions

    #     Parameters
    #     ----------
    #     z_basal : tensor
    #         Basal states
    #     """
    #     pred_treatments = self.treatments_adv_nn(z_basal)
    #     pred_covariates = dict()
    #     for cat_cov_name in self.module.cat_to_ncats:
    #         pred_covariates[cat_cov_name] = self.covariates_adv_nn[cat_cov_name](
    #             z_basal
    #         )
    #     return pred_treatments, pred_covariates

    # def adversarial_losses(self, tensors, inference_outputs, generative_outputs):
    #     """Computes adversarial classification losses and regularizations"""
    #     z_basal = inference_outputs["z_basal"]
    #     treatments = tensors["treatments"]
    #     c_dict = inference_outputs["c_dict"]
    #     pred_treatments, pred_covariates = self._adversarial_classifications(z_basal)

    #     # Classification losses
    #     adv_cats_loss = 0.0
    #     for cat_cov_name in self.module.cat_to_ncats:
    #         adv_cats_loss += self.adv_loss_covariates(
    #             pred_covariates[cat_cov_name],
    #             c_dict[cat_cov_name].long().squeeze(-1),
    #         )
    #     adv_t_loss = self.adv_loss_treatments(pred_treatments, (treatments > 0).float())
    #     adv_loss = adv_t_loss + adv_cats_loss

    #     # Penalty losses
    #     adv_penalty_cats = 0.0
    #     for cat_cov_name in self.module.cat_to_ncats:
    #         cat_penalty = (
    #             torch.autograd.grad(
    #                 pred_covariates[cat_cov_name].sum(), z_basal, create_graph=True
    #             )[0]
    #             .pow(2)
    #             .mean()
    #         )
    #         adv_penalty_cats += cat_penalty

    #     adv_penalty_treatments = (
    #         torch.autograd.grad(
    #             pred_treatments.sum(),
    #             z_basal,
    #             create_graph=True,
    #         )[0]
    #         .pow(2)
    #         .mean()
    #     )
    #     adv_penalty = adv_penalty_cats + adv_penalty_treatments

    #     return dict(
    #         adv_loss=adv_loss,
    #         adv_penalty=adv_penalty,
    #     )

    def configure_optimizers(self):
        optimizer_autoencoder = torch.optim.Adam(
            list(self.module.encoder.parameters()) +
            list(self.module.decoder.parameters()) +
            list(self.module.drug_network.drug_embedding.parameters()) +
            list(self.module.covars_embedding.parameters()),
            lr=self.autoencoder_lr,
            weight_decay=self.autoencoder_wd)

        optimizer_adversaries = torch.optim.Adam(
            list(self.module.drugs_classifier.parameters()) +
            list(self.module.covars_classifiers.parameters()),
            lr=self.adversary_lr,
            weight_decay=self.adversary_wd)

        optimizer_dosers = torch.optim.Adam(
            self.module.drug_network.dosers.parameters(),
            lr=self.dosers_lr,
            weight_decay=self.dosers_wd)

        # params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        # optimizer1 = torch.optim.Adam(
        #     params1, lr=self.lr, eps=self.autoencoder_wd, weight_decay=self.weight_decay
        # )
        # params2 = filter(
        #     lambda p: p.requires_grad,
        #     list(self.covariates_adv_nn.parameters())
        #     + list(self.treatments_adv_nn.parameters()),
        # )
        # optimizer2 = torch.optim.Adam(
        #     params2,
        #     lr=self.adversary_lr,
        #     eps=self.adversary_wd,
        #     weight_decay=self.weight_decay,
        # )
        # optims = [optimizer1, optimizer2]
        
        optimizers = [optimizer_autoencoder, optimizer_adversaries, optimizer_dosers]
        if self.step_size_lr is not None:
            scheduler_autoencoder = StepLR(optimizer_autoencoder, step_size=self.step_size_lr)
            scheduler_adversaries = StepLR(optimizer_adversaries, step_size=self.step_size_lr)
            scheduler_dosers = StepLR(optimizer_dosers, step_size=self.step_size_lr)
            schedulers = [scheduler_autoencoder, scheduler_adversaries, scheduler_dosers]
            return optimizers, schedulers
        else:
            return optimizers

    def training_step(self, batch, batch_idx):
        opt, opt_adv, opt_dosers = self.optimizers()

        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False)
        reconstruction_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )

        adv_results = self.module.adversarial_loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        
        # Adversarial update
        if self.iter_count % self.adversary_steps:
            opt_adv.zero_grad()
            self.manual_backward(adv_results['adv_loss'] + self.penalty_adversary * adv_results['penalty_adv'])
            opt_adv.step()

        # Model update
        else:
            opt.zero_grad()
            opt_dosers.zero_grad()
            self.manual_backward(reconstruction_loss - self.reg_adversary * adv_results['adv_loss'])
            opt.step()
            opt_adv.step()

        self.iter_count += 1
        self.log("recon_loss", reconstruction_loss.item(), on_step=True, prog_bar=True)
        self.log("adv_loss", adv_results['adv_loss'], on_step=True, prog_bar=True)
        self.log("penalty_adv", adv_results['penalty_adv'], on_step=True, prog_bar=True)

        for key, val in adv_results.items():
            adv_results[key] = val.item()

        adv_results.update({'recon_loss': reconstruction_loss.item()})

        return adv_results
        
    def training_epoch_end(self, outputs):
        keys = ['recon_loss', 'adv_loss', 'penalty_adv', 'adv_drugs', 'penalty_drugs']
        for key in keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))

        for covar in self.covars_to_ncovars.keys():
            key1, key2 = f'adv_{covar}', f'penalty_{covar}'
            self.epoch_history[key1].append(np.mean([output[key1] for output in outputs]))
            self.epoch_history[key2].append(np.mean([output[key2] for output in outputs]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('train')
        
        if self.step_size_lr:
            sch, sch_adv, sch_dosers = self.lr_schedulers()
            sch.step()
            sch_adv.step()
            sch_dosers.step()

    def validation_step(self, batch, batch_idx):
        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False)
        reconstruction_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )
        adv_results = self.module.adversarial_loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )

        for key, val in adv_results.items():
            adv_results[key] = val.item()
            
        adv_results.update({'recon_loss': reconstruction_loss.item()})

        return adv_results

    def validation_epoch_end(self, outputs):
        keys = ['recon_loss', 'adv_loss', 'penalty_adv', 'adv_drugs', 'penalty_drugs']
        for key in keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))

        for covar in self.covars_to_ncovars.keys():
            key1, key2 = f'adv_{covar}', f'penalty_{covar}'
            self.epoch_history[key1].append(np.mean([output[key1] for output in outputs]))
            self.epoch_history[key2].append(np.mean([output[key2] for output in outputs]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('valid')

        self.log('reconstruction_loss_validation', self.epoch_history['recon_loss'][-1])

    def on_validation_epoch_start(self) -> None:
        self.module.train()
        torch.set_grad_enabled(True)

    def on_validation_epoch_end(self) -> None:
        self.zero_grad()