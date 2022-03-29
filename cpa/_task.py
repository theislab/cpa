from typing import Union

import torch
from torch.optim.lr_scheduler import StepLR

from scvi.module.base import BaseModuleClass
from scvi.train import TrainingPlan

import numpy as np


class CPATrainingPlan(TrainingPlan):
    def __init__(
            self,
            module: BaseModuleClass,
            covars_to_ncovars: dict,
            autoencoder_lr=3e-4,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            n_epochs_warmup: Union[int, None] = None,
            adversary_steps: int = 3,
            reg_adversary: float = 60,
            penalty_adversary: float = 60,
            dosers_lr=4e-3,
            dosers_wd=1e-7,
            kl_weight=None,
            adversary_lr=3e-4,
            adversary_wd=4e-7,
            autoencoder_wd=4e-7,
            step_size_lr: int = 45,
            batch_size: int = 256,
    ):
        """Training plan for the CPA model"""
        super().__init__(
            module=module,
            lr=autoencoder_lr,
            weight_decay=autoencoder_wd,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=False,
            lr_factor=None,
            lr_patience=None,
            lr_threshold=None,
            lr_scheduler_metric=None,
            lr_min=None,
        )

        self.n_epochs_warmup = n_epochs_warmup if n_epochs_warmup is not None else 0

        self.covars_encoder = covars_to_ncovars

        self.autoencoder_wd = autoencoder_wd
        self.autoencoder_lr = autoencoder_lr

        self.adversary_lr = adversary_lr
        self.adversary_wd = adversary_wd
        self.adversary_steps = adversary_steps
        self.reg_adversary = reg_adversary
        self.penalty_adversary = penalty_adversary

        self.kl_coeff = kl_weight

        self.dosers_lr = dosers_lr
        self.dosers_wd = dosers_wd

        self.step_size_lr = step_size_lr

        self.batch_size = batch_size

        self.automatic_optimization = False
        self.iter_count = 0

        self.epoch_history = {
            'mode': [],
            'epoch': [],
            'recon_loss': [],
            'adv_loss': [],
            'penalty_adv': [],
            'adv_drugs': [],
            'penalty_drugs': [],
            'reg_mean': [],
            'reg_var': [],
            'disent_basal': [],
            'disent_after': [],
        }

        for covar in self.covars_encoder.keys():
            self.epoch_history[f'adv_{covar}'] = []
            self.epoch_history[f'penalty_{covar}'] = []

    def configure_optimizers(self):
        optimizer_autoencoder = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.module.encoder.parameters())) +
            list(filter(lambda p: p.requires_grad, self.module.decoder.parameters())) +
            list(filter(lambda p: p.requires_grad, self.module.drug_network.drug_embedding.parameters())) +
            list(filter(lambda p: p.requires_grad, self.module.covars_embedding.parameters())),
            lr=self.autoencoder_lr,
            weight_decay=self.autoencoder_wd)

        optimizer_adversaries = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.module.drugs_classifier.parameters())) +
            list(filter(lambda p: p.requires_grad, self.module.covars_classifiers.parameters())),
            lr=self.adversary_lr,
            weight_decay=self.adversary_wd)

        optimizer_dosers = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.module.drug_network.dosers.parameters()),
            lr=self.dosers_lr,
            weight_decay=self.dosers_wd)

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

        if self.current_epoch >= self.n_epochs_warmup:
            # Adversarial update
            if self.iter_count % self.adversary_steps != 0:
                opt_adv.zero_grad()
                latent_basal = inf_outputs['latent_basal']
                adv_results = self.module.adversarial_loss(tensors=batch, latent_basal=latent_basal)
                self.manual_backward(adv_results['adv_loss'] + self.penalty_adversary * adv_results['penalty_adv'])
                opt_adv.step()

                for key, val in adv_results.items():
                    adv_results[key] = val.item()

                results = adv_results.copy()
                results.update({'recon_loss': 0.0})

            # Model update
            else:
                opt.zero_grad()
                opt_dosers.zero_grad()
                reconstruction_loss, kl_loss = self.module.loss(
                    tensors=batch,
                    inference_outputs=inf_outputs,
                    generative_outputs=gen_outputs,
                )
                latent_basal = inf_outputs['latent_basal']
                adv_results = self.module.adversarial_loss(tensors=batch, latent_basal=latent_basal)

                loss = reconstruction_loss - self.reg_adversary * adv_results['adv_loss']
                if self.kl_coeff is not None and self.kl_coeff != 0.0:
                    loss += self.kl_coeff * kl_loss.mean()
                self.manual_backward(loss)
                opt.step()
                opt_dosers.step()

                for key, val in adv_results.items():
                    adv_results[key] = val.item()

                results = adv_results.copy()

                results.update({'recon_loss': reconstruction_loss.item()})

        else:
            adv_results = {'adv_loss': 0.0, 'adv_drugs': 0.0, 'penalty_adv': 0.0, 'penalty_drugs': 0.0}
            for covar in self.covars_encoder.keys():
                adv_results[f'adv_{covar}'] = 0.0
                adv_results[f'penalty_{covar}'] = 0.0

            results = adv_results.copy()

            opt.zero_grad()
            opt_dosers.zero_grad()
            reconstruction_loss, kl_loss = self.module.loss(
                tensors=batch,
                inference_outputs=inf_outputs,
                generative_outputs=gen_outputs,
            )
            loss = reconstruction_loss
            if self.kl_coeff is not None:
                loss += self.kl_coeff * kl_loss
            self.manual_backward(loss)
            opt.step()
            opt_dosers.step()

            results.update({'recon_loss': reconstruction_loss.item()})

        self.iter_count += 1

        results.update({'reg_mean': 0.0, 'reg_var': 0.0})
        results.update({'disent_basal': 0.0})
        results.update({'disent_after': 0.0})

        return results

    def training_epoch_end(self, outputs):
        keys = ['recon_loss', 'adv_loss', 'penalty_adv', 'adv_drugs', 'penalty_drugs', 'reg_mean', 'reg_var',
                'disent_basal', 'disent_after']
        for key in keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))

        for covar in self.covars_encoder.keys():
            key1, key2 = f'adv_{covar}', f'penalty_{covar}'
            self.epoch_history[key1].append(np.mean([output[key1] for output in outputs]))
            self.epoch_history[key2].append(np.mean([output[key2] for output in outputs]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('train')

        self.log("recon", self.epoch_history['recon_loss'][-1], prog_bar=True)
        self.log("adv_loss", self.epoch_history['adv_loss'][-1], prog_bar=True)
        # self.log("penalty_adv", self.epoch_history['penalty_adv'][-1], prog_bar=True)
        # self.log("reg_mean", self.epoch_history['reg_mean'][-1], prog_bar=True)
        # self.log("reg_var", self.epoch_history['reg_var'][-1], prog_bar=True)
        # self.log("disent_drugs", self.epoch_history['disent_drugs'][-1], prog_bar=True)

        if self.current_epoch > 1 and self.current_epoch % self.step_size_lr == 0:
            sch, sch_adv, sch_dosers = self.lr_schedulers()
            sch.step()
            sch_adv.step()
            sch_dosers.step()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('v_num')
        # items.pop('loss')
        return items

    def validation_step(self, batch, batch_idx):
        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False)

        reconstruction_loss, kl_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )

        # if self.current_epoch >= self.n_epochs_warmup:
        #     adv_results = self.module.adversarial_loss(
        #         tensors=batch,
        #         inference_outputs=inf_outputs,
        #         generative_outputs=gen_outputs,
        #     )
        #     for key, val in adv_results.items():
        #         adv_results[key] = val.item()
        # else:
        adv_results = {'adv_loss': 0.0, 'adv_drugs': 0.0, 'penalty_adv': 0.0, 'penalty_drugs': 0.0}
        for covar in self.covars_encoder.keys():
            adv_results[f'adv_{covar}'] = 0.0
            adv_results[f'penalty_{covar}'] = 0.0

        r2_mean, r2_var = self.module.r2_metric(batch, inf_outputs, gen_outputs)
        disent_basal, disent_after = self.module.disentanglement(batch, inf_outputs, gen_outputs)

        results = adv_results
        results.update({'reg_mean': r2_mean, 'reg_var': r2_var})
        results.update({'disent_basal': disent_basal})
        results.update({'disent_after': disent_after})
        results.update({'recon_loss': reconstruction_loss.item()})
        results.update({'cpa_metric': r2_mean + 1.0 + len(
            [covar for covar, unique_covars in self.covars_encoder.items() if
             len(unique_covars) > 1]) - disent_basal + disent_after})

        return results

    def validation_epoch_end(self, outputs):
        keys = ['recon_loss', 'adv_loss', 'penalty_adv', 'adv_drugs', 'penalty_drugs', 'reg_mean', 'reg_var',
                'disent_basal', 'disent_after']
        for key in keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))

        for covar in self.covars_encoder.keys():
            key1, key2 = f'adv_{covar}', f'penalty_{covar}'
            self.epoch_history[key1].append(np.mean([output[key1] for output in outputs]))
            self.epoch_history[key2].append(np.mean([output[key2] for output in outputs]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('valid')

        self.log('val_recon', self.epoch_history['recon_loss'][-1], prog_bar=True)
        self.log('cpa_metric', np.mean([output['cpa_metric'] for output in outputs]), prog_bar=False)
        self.log('val_reg_mean', self.epoch_history['reg_mean'][-1], prog_bar=False)
        self.log('val_disnt_basal', self.epoch_history['disent_basal'][-1], prog_bar=True)
        self.log('val_disnt_after', self.epoch_history['disent_after'][-1], prog_bar=True)
        # self.log('val_reg_var', self.epoch_history['reg_var'][-1], prog_bar=True)
