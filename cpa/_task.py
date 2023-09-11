import math

from collections import defaultdict
from typing import Union

import torch
from scvi.module import Classifier
from torch import nn
from torch.optim.lr_scheduler import StepLR

from scvi.train import TrainingPlan

import numpy as np
from torchmetrics.functional import accuracy

from ._module import CPAModule
from ._utils import CPA_REGISTRY_KEYS, FocalLoss
from typing import Optional


class CPATrainingPlan(TrainingPlan):
    def __init__(
            self,
            module: CPAModule,
            covars_to_ncovars: dict,
            n_adv_perts: int,
            lr=5e-4,
            wd=1e-6,
            n_steps_pretrain_ae: Optional[int] = None,
            n_epochs_pretrain_ae: Optional[int] = None,
            n_steps_kl_warmup: Optional[int] = None,
            n_epochs_kl_warmup: Optional[int] = None,
            n_steps_adv_warmup: Optional[int] = None,
            n_epochs_adv_warmup: Optional[int] = None,
            n_epochs_mixup_warmup: Optional[int] = None,
            n_epochs_verbose: Optional[int] = 10,
            mixup_alpha: float = 0.0,
            adv_steps: int = 3,
            reg_adv: float = 1.,
            pen_adv: float = 1.,
            n_hidden_adv: int = 64,
            n_layers_adv: int = 3,
            use_batch_norm_adv: bool = True,
            use_layer_norm_adv: bool = False,
            dropout_rate_adv: float = 0.1,
            adv_lr=3e-4,
            adv_wd=4e-7,
            doser_lr=3e-4,
            doser_wd=4e-7,
            step_size_lr: Optional[int] = 45,
            do_clip_grad: Optional[bool] = False,
            gradient_clip_value: Optional[float] = 3.0,
            drug_weights: Optional[list] = None,
            adv_loss: Optional[str] = 'cce',
    ):
        """
        Training plan for the CPA model. 

        Parameters
        ----------
        module: CPAModule
            The model to train.
        covars_to_ncovars: dict
            Dictionary of covariates to their unique values.
        n_adv_perts: int
            Number of all possible perturbations (single/combinatorial).
        lr: float
            Learning rate for the autoencoder.
        wd: float
            Weight decay for the autoencoder.
        n_steps_pretrain_ae: Optional[int]
            Number of steps to pretrain the autoencoder.
        n_epochs_pretrain_ae: Optional[int]
            Number of epochs to pretrain the autoencoder.
        n_steps_kl_warmup: Optional[int]
            Number of steps to warmup the KL term. Will be effective if `variational` has been set True for the model.
        n_epochs_kl_warmup: Optional[int]
            Number of epochs to warmup the KL term. Will be effective if `variational` has been set True for the model.
        n_steps_adv_warmup: Optional[int]
            Number of steps to warmup the adversarial term.
        n_epochs_adv_warmup: Optional[int]
            Number of epochs to warmup the adversarial term.
        n_epochs_mixup_warmup: Optional[int]
            Number of epochs to warmup the mixup term.
        n_epochs_verbose: Optional[int]
            Number of epochs to print the training progress.
        mixup_alpha: float
            Alpha parameter for the mixup term. set this to 0.0 to disable mixup.
        adv_steps: int
            Number of steps to train the adversarial term.
        reg_adv: float
            Regularization parameter for the adversarial term.
        pen_adv: float
            Penalty parameter for the adversarial term.
        n_hidden_adv: int
            Number of hidden units for the adversarial classifier.
        n_layers_adv: int
            Number of layers for the adversarial classifier.
        use_batch_norm_adv: bool
            Whether to use batch normalization for the adversarial classifier.
        use_layer_norm_adv: bool
            Whether to use layer normalization for the adversarial classifier.
        dropout_rate_adv: float
            Dropout rate for the adversarial classifier.
        adv_lr: float
            Learning rate for the adversarial classifier.
        adv_wd: float
            Weight decay for the adversarial classifier.
        doser_lr: float
            Learning rate for the dosers.
        doser_wd: float
            Weight decay for the dosers.
        step_size_lr: Optional[int]
            Step size for the learning rate scheduler.
        do_clip_grad: Optional[bool]
            Whether to clip the gradients.
        gradient_clip_value: Optional[float]
            Value to clip the gradients. Will be effective if `do_clip_grad` is True.
        drug_weights: Optional[list]
            Weights for the perturbations to be used in the adversarial loss.
        adv_loss: Optional[str]
            Adversarial loss to be used. Can be either 'cce' or 'focal'.
        """
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=wd,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=False,
            lr_factor=None,
            lr_patience=None,
            lr_threshold=None,
            lr_scheduler_metric=None,
            lr_min=None,
        )

        self.automatic_optimization = False

        self.wd = wd

        self.covars_encoder = covars_to_ncovars

        self.mixup_alpha = mixup_alpha
        self.n_epochs_mixup_warmup = n_epochs_mixup_warmup

        self.n_steps_pretrain_ae = n_steps_pretrain_ae
        self.n_epochs_pretrain_ae = n_epochs_pretrain_ae

        self.n_steps_adv_warmup = n_steps_adv_warmup
        self.n_epochs_adv_warmup = n_epochs_adv_warmup

        self.n_epochs_verbose = n_epochs_verbose

        self.adv_steps = adv_steps

        self.reg_adv = reg_adv
        self.pen_adv = pen_adv

        self.adv_lr = adv_lr
        self.adv_wd = adv_wd

        self.doser_lr = doser_lr
        self.doser_wd = doser_wd

        self.step_size_lr = step_size_lr

        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value

        self.metrics = ['recon_loss', 'KL',
                        'disnt_basal', 'disnt_after',
                        'r2_mean', 'r2_var',
                        'adv_loss', 'penalty_adv', 'adv_perts', 'acc_perts', 'penalty_perts']

        self.epoch_history = defaultdict(list)
        self.n_adv_perts = n_adv_perts

        self.perturbation_classifier = Classifier(
            n_input=self.module.n_latent,
            n_labels=n_adv_perts,
            n_hidden=n_hidden_adv,
            n_layers=n_layers_adv,
            use_batch_norm=use_batch_norm_adv,
            use_layer_norm=use_layer_norm_adv,
            dropout_rate=dropout_rate_adv,
            activation_fn=nn.ReLU,
            logits=True,
        )

        self.covars_classifiers = nn.ModuleDict(
            {
                key: Classifier(n_input=self.module.n_latent,
                                n_labels=len(unique_covars),
                                n_hidden=n_hidden_adv,
                                n_layers=n_layers_adv,
                                use_batch_norm=use_batch_norm_adv,
                                use_layer_norm=use_layer_norm_adv,
                                dropout_rate=dropout_rate_adv,
                                logits=True)
                if len(unique_covars) > 1 else None

                for key, unique_covars in self.covars_encoder.items()
            }
        )

        self.drug_weights = torch.tensor(drug_weights).to(self.device) if drug_weights else torch.ones(
            self.n_adv_perts).to(self.device)

        self.adv_loss = adv_loss.lower()
        self.gamma = 2.0
        if self.adv_loss == 'cce':
            self.adv_loss_drugs = nn.CrossEntropyLoss(weight=self.drug_weights)
            self.adv_loss_fn = nn.CrossEntropyLoss()
        elif self.adv_loss == 'focal':
            self.adv_loss_drugs = FocalLoss(alpha=self.drug_weights, gamma=self.gamma, reduction='mean')
            self.adv_loss_fn = FocalLoss(gamma=self.gamma, reduction='mean')
        
    @property
    def adv_lambda(self):
        slope = self.reg_adv
        if self.n_steps_adv_warmup:
            global_step = self.global_step

            if self.n_steps_pretrain_ae:
                 global_step -= self.n_steps_pretrain_ae

            if global_step <= self.n_steps_adv_warmup:
                proportion = global_step / self.n_steps_adv_warmup
                return slope * proportion
            else:
                return slope
        elif self.n_epochs_adv_warmup is not None:
            current_epoch = self.current_epoch

            if self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            if current_epoch <= self.n_epochs_adv_warmup:
                proportion = current_epoch / self.n_epochs_adv_warmup
                return slope * proportion
            else:
                return slope
        else:
            return slope

    @property
    def alpha_mixup(self):
        if self.n_epochs_mixup_warmup:
            current_epoch = self.current_epoch

            if self.n_epochs_pretrain_ae:
                current_epoch -= self.current_epoch

            if current_epoch <= self.n_epochs_mixup_warmup:
                proportion = current_epoch / self.n_epochs_mixup_warmup

                return self.mixup_alpha * proportion
            else:
                return self.mixup_alpha
        else:
            return self.mixup_alpha

    @property
    def do_start_adv_training(self):
        if self.n_steps_pretrain_ae:
            return self.global_step > self.n_steps_pretrain_ae
        elif self.n_epochs_pretrain_ae:
            return self.current_epoch > self.n_epochs_pretrain_ae
        else:
            return True

    def adversarial_loss(self, tensors, z_basal, mixup_lambda: float = 1.0, compute_penalty=True):
        """Computes adversarial classification losses and regularizations"""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        covars_dict = dict()
        for covar, unique_covars in self.covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  # (batch_size,)
            covars_dict[covar] = encoded_covars

        covars_pred = {}
        for covar in self.covars_encoder.keys():
            if self.covars_classifiers[covar] is not None:
                covar_pred = self.covars_classifiers[covar](z_basal)
                covars_pred[covar] = covar_pred
            else:
                covars_pred[covar] = None

        adv_results = {}

        # Classification losses for different covariates
        for covar, covars in self.covars_encoder.items():
            adv_results[f'adv_{covar}'] = mixup_lambda * self.adv_loss_fn(
                covars_pred[covar],
                covars_dict[covar].long(),
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device) + (
                    1. - mixup_lambda) * self.adv_loss_fn(
                covars_pred[covar],
                covars_dict[covar + '_mixup'].long(),
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)
            adv_results[f'acc_{covar}'] = accuracy(
                covars_pred[covar].argmax(1), covars_dict[covar].long(), task='multiclass',
                num_classes=len(covars)) \
                if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        adv_results['adv_loss'] = sum([adv_results[f'adv_{key}'] for key in self.covars_encoder.keys()])

        perturbations = tensors[CPA_REGISTRY_KEYS.PERTURBATION_KEY].view(-1, )
        perturbations_mixup = tensors[CPA_REGISTRY_KEYS.PERTURBATION_KEY + '_mixup'].view(-1, )

        perturbations_pred = self.perturbation_classifier(z_basal)

        adv_results['adv_perts'] = mixup_lambda * self.adv_loss_drugs(perturbations_pred,
                                                                      perturbations.long()) + (
                                           1. - mixup_lambda) * self.adv_loss_drugs(perturbations_pred,
                                                                                    perturbations_mixup.long())

        adv_results['acc_perts'] = mixup_lambda * accuracy(
            perturbations_pred.argmax(1), perturbations.long().view(-1, ), average='macro',
            num_classes=self.n_adv_perts, task='multiclass',
        ) + (1. - mixup_lambda) * accuracy(
            perturbations_pred.argmax(1), perturbations_mixup.long().view(-1, ), average='macro',
            num_classes=self.n_adv_perts, task='multiclass',
        )

        adv_results['adv_loss'] += adv_results['adv_perts']

        if compute_penalty:
            # Penalty losses
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = (
                    torch.autograd.grad(
                        covars_pred[covar].sum(),
                        z_basal,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0].pow(2).mean()
                ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_adv'] = sum([adv_results[f'penalty_{covar}'] for covar in self.covars_encoder.keys()])

            adv_results['penalty_perts'] = (
                torch.autograd.grad(
                    perturbations_pred.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].pow(2).mean()
            )

            adv_results['penalty_adv'] += adv_results['penalty_perts']
        else:
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_perts'] = torch.as_tensor(0.0).to(self.device)
            adv_results['penalty_adv'] = torch.as_tensor(0.0).to(self.device)

        return adv_results
    
    def configure_optimizers(self):
        ae_params = list(filter(lambda p: p.requires_grad, self.module.encoder.parameters())) + \
                    list(filter(lambda p: p.requires_grad, self.module.decoder.parameters())) + \
                    list(filter(lambda p: p.requires_grad, self.module.pert_network.pert_embedding.parameters())) + \
                    list(filter(lambda p: p.requires_grad, self.module.covars_embeddings.parameters()))

        if self.module.recon_loss in ['zinb', 'nb']:
            ae_params += [self.module.px_r]

        optimizer_autoencoder = torch.optim.Adam(
            ae_params,
            lr=self.lr,
            weight_decay=self.wd)

        scheduler_autoencoder = StepLR(optimizer_autoencoder, step_size=self.step_size_lr, gamma=0.9)

        doser_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.dosers.parameters()))
        optimizer_doser = torch.optim.Adam(
            doser_params, lr=self.doser_lr, weight_decay=self.doser_wd,
        )
        scheduler_doser = StepLR(optimizer_doser, step_size=self.step_size_lr, gamma=0.9)

        adv_params = list(filter(lambda p: p.requires_grad, self.perturbation_classifier.parameters())) + \
                     list(filter(lambda p: p.requires_grad, self.covars_classifiers.parameters()))

        optimizer_adversaries = torch.optim.Adam(
            adv_params,
            lr=self.adv_lr,
            weight_decay=self.adv_wd)
        scheduler_adversaries = StepLR(optimizer_adversaries, step_size=self.step_size_lr, gamma=0.9)

        optimizers = [optimizer_autoencoder, optimizer_doser, optimizer_adversaries]
        schedulers = [scheduler_autoencoder, scheduler_doser, scheduler_adversaries]

        if self.step_size_lr is not None:
            return optimizers, schedulers
        else:
            return optimizers

    def training_step(self, batch, batch_idx):
        opt, opt_doser, opt_adv = self.optimizers()

        mixup_alpha = self.alpha_mixup

        batch, mixup_lambda = self.module.mixup_data(batch, alpha=mixup_alpha)

        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False,
                                                       inference_kwargs={
                                                           'mixup_lambda': mixup_lambda,
                                                       })

        recon_loss, kl_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )

        if self.do_start_adv_training:
            if self.adv_steps is None:
                opt.zero_grad()
                opt_doser.zero_grad()

                z_basal = inf_outputs['z_basal']

                adv_results = self.adversarial_loss(tensors=batch,
                                                    z_basal=z_basal,
                                                    mixup_lambda=mixup_lambda,
                                                    compute_penalty=False)

                loss = recon_loss + self.kl_weight * kl_loss - self.adv_lambda * adv_results['adv_loss']

                self.manual_backward(loss)

                if self.do_clip_grad:
                    self.clip_gradients(opt,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")
                    self.clip_gradients(opt_doser,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")
                opt.step()
                opt_doser.step()

                opt_adv.zero_grad()

                adv_results = self.adversarial_loss(tensors=batch,
                                                    z_basal=z_basal.detach(),
                                                    mixup_lambda=mixup_lambda,
                                                    compute_penalty=True)

                adv_loss = adv_results['adv_loss'] + self.pen_adv * adv_results['penalty_adv']

                self.manual_backward(adv_loss)

                if self.do_clip_grad:
                    self.clip_gradients(opt_adv,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")

                opt_adv.step()

            elif batch_idx % self.adv_steps != 0:
                opt_adv.zero_grad()

                z_basal = inf_outputs['z_basal']

                adv_results = self.adversarial_loss(tensors=batch,
                                                    z_basal=z_basal.detach(),
                                                    mixup_lambda=mixup_lambda,
                                                    compute_penalty=True)

                adv_loss = adv_results['adv_loss'] + self.pen_adv * adv_results['penalty_adv']

                self.manual_backward(adv_loss)

                if self.do_clip_grad:
                    self.clip_gradients(opt_adv,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")

                opt_adv.step()

            # Model update
            else:
                opt.zero_grad()
                opt_doser.zero_grad()

                z_basal = inf_outputs['z_basal']

                adv_results = self.adversarial_loss(tensors=batch,
                                                    z_basal=z_basal,
                                                    mixup_lambda=mixup_lambda,
                                                    compute_penalty=False)

                loss = recon_loss + self.kl_weight * kl_loss - self.adv_lambda * adv_results['adv_loss']

                self.manual_backward(loss)

                if self.do_clip_grad:
                    self.clip_gradients(opt,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")
                    self.clip_gradients(opt_doser,
                                        gradient_clip_val=self.gradient_clip_value,
                                        gradient_clip_algorithm="norm")
                opt.step()
                opt_doser.step()

        else:
            opt.zero_grad()
            opt_doser.zero_grad()

            z_basal = inf_outputs['z_basal']

            loss = recon_loss + self.kl_weight * kl_loss

            self.manual_backward(loss)

            if self.do_clip_grad:
                self.clip_gradients(opt,
                                    gradient_clip_val=self.gradient_clip_value,
                                    gradient_clip_algorithm="norm")
                self.clip_gradients(opt_doser,
                                    gradient_clip_val=self.gradient_clip_value,
                                    gradient_clip_algorithm="norm")

            opt.step()
            opt_doser.step()

            opt_adv.zero_grad()

            adv_results = self.adversarial_loss(tensors=batch,
                                                z_basal=z_basal.detach(),
                                                mixup_lambda=mixup_lambda,
                                                compute_penalty=True)

            adv_loss = adv_results['adv_loss'] + self.pen_adv * adv_results['penalty_adv']

            self.manual_backward(adv_loss)

            if self.do_clip_grad:
                self.clip_gradients(opt_adv,
                                    gradient_clip_val=self.gradient_clip_value,
                                    gradient_clip_algorithm="norm")

            opt_adv.step()

        r2_mean, r2_var = self.module.r2_metric(batch, inf_outputs, gen_outputs, mode='direct')

        for key, val in adv_results.items():
            adv_results[key] = val.item()

        results = adv_results.copy()
        results.update({'recon_loss': recon_loss.item()})
        results.update({'KL': kl_loss.item()})

        results.update({'r2_mean': r2_mean, 'r2_var': r2_var})
        results.update({'r2_mean_lfc': 0.0, 'r2_var_lfc': 0.0})
        results.update({'cpa_metric': 0.0})
        results.update({'disnt_basal': 0.0, 'disnt_after': 0.0})

        return results

    def training_epoch_end(self, outputs):
        for key in self.metrics:
            if key in ['disnt_basal', 'disnt_after']:
                self.epoch_history[key].append(0.0)
            else:
                self.epoch_history[key].append(np.mean([output[key] for output in outputs if output[key] != 0.0]))

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                key1, key2, key3 = f'adv_{covar}', f'penalty_{covar}', f'acc_{covar}'
                self.epoch_history[key1].append(np.mean([output[key1] for output in outputs if output[key1] != 0.0]))
                self.epoch_history[key2].append(np.mean([output[key2] for output in outputs if output[key2] != 0.0]))
                self.epoch_history[key3].append(np.mean([output[key3] for output in outputs if output[key3] != 0.0]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('train')

        self.log("recon", self.epoch_history['recon_loss'][-1], prog_bar=True)
        self.log("r2_mean", self.epoch_history['r2_mean'][-1], prog_bar=True)
        self.log("adv_loss", self.epoch_history['adv_loss'][-1], prog_bar=True)
        self.log("acc_pert", self.epoch_history['acc_perts'][-1], prog_bar=True)
        for covar, nc in self.covars_encoder.items():
            if len(nc) > 1:
                self.log(f'acc_{covar}', self.epoch_history[f'acc_{covar}'][-1], prog_bar=True)

        if self.current_epoch > 1 and self.current_epoch % self.step_size_lr == 0:
            sch, sch_doser, sch_adv = self.lr_schedulers()
            sch.step()
            sch_doser.step()
            sch_adv.step()

    def validation_step(self, batch, batch_idx):
        batch, mixup_lambda = self.module.mixup_data(batch, alpha=0.0)  # No mixup during validation

        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False,
                                                       inference_kwargs={
                                                           'mixup_lambda': 1.0,
                                                       })

        recon_loss, kl_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
        )

        adv_results = {'adv_loss': 0.0, 'cycle_loss': 0.0, 'penalty_adv': 0.0,
                       'adv_perts': 0.0, 'acc_perts': 0.0, 'penalty_perts': 0.0}
        for covar in self.covars_encoder.keys():
            adv_results[f'adv_{covar}'] = 0.0
            adv_results[f'acc_{covar}'] = 0.0
            adv_results[f'penalty_{covar}'] = 0.0

        r2_mean, r2_var = self.module.r2_metric(batch, inf_outputs, gen_outputs, mode='direct')
        disnt_basal, disnt_after = self.module.disentanglement(batch, inf_outputs, gen_outputs)

        results = adv_results
        results.update({'r2_mean': r2_mean, 'r2_var': r2_var})
        results.update({'disnt_basal': disnt_basal})
        results.update({'disnt_after': disnt_after})
        results.update({'KL': kl_loss.item()})
        results.update({'recon_loss': recon_loss.item()})
        results.update({'cpa_metric': r2_mean + 0.5 * r2_var + math.e ** (disnt_after - disnt_basal)})

        return results

    def validation_epoch_end(self, outputs):
        for key in self.metrics:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs if output[key] != 0.0]))

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                key1, key2, key3 = f'adv_{covar}', f'penalty_{covar}', f'acc_{covar}'
                self.epoch_history[key1].append(np.mean([output[key1] for output in outputs if output[key1] != 0.0]))
                self.epoch_history[key2].append(np.mean([output[key2] for output in outputs if output[key2] != 0.0]))
                self.epoch_history[key3].append(np.mean([output[key3] for output in outputs if output[key3] != 0.0]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('valid')

        self.log('val_recon', self.epoch_history['recon_loss'][-1], prog_bar=True)
        self.log('cpa_metric', np.mean([output['cpa_metric'] for output in outputs]), prog_bar=False)
        self.log('disnt_basal', self.epoch_history['disnt_basal'][-1], prog_bar=True)
        self.log('disnt_after', self.epoch_history['disnt_after'][-1], prog_bar=True)
        self.log('val_r2_mean', self.epoch_history['r2_mean'][-1], prog_bar=True)
        self.log('val_r2_var', self.epoch_history['r2_var'][-1], prog_bar=False)
        self.log('val_KL', self.epoch_history['KL'][-1], prog_bar=True)

        if self.current_epoch % self.n_epochs_verbose == self.n_epochs_verbose - 1:
            print(f'\ndisnt_basal = {self.epoch_history["disnt_basal"][-1]}')
            print(f'disnt_after = {self.epoch_history["disnt_after"][-1]}')
            print(f'val_r2_mean = {self.epoch_history["r2_mean"][-1]}')
            print(f'val_r2_var = {self.epoch_history["r2_var"][-1]}')

