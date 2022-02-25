import logging
import os
from typing import Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from anndata import AnnData
from scvi.data import setup_anndata
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.train._callbacks import SaveBestState
from scvi.utils import setup_anndata_dsp

from ._module import CPAModule, _CE_CONSTANTS
from ._task import CPATrainingPlan
from ._utils import ManualDataSplitter

logger = logging.getLogger(__name__)


class CPA(BaseModelClass):
    def __init__(
            self,
            adata: AnnData,
            n_latent: int,
            loss_ae: str,
            doser_type: str,
            covars_encoder: dict,
            split_key: str = None,
            drug_encoder=None,
            **hyper_params,
    ):
        """CPA model

        Parameters
        ----------
        adata : Anndata
            Must be properly registered

        batch_keys_to_dim : str
        Map between covariates and the associated number of categories

        split_key : str, optional
            Key used to split the data between train test and validation.
            This must correspond to a observation key for the adata, composed of values
            'train', 'test', and 'ood'. By default None
        """
        super().__init__(adata)
        self.n_genes = self.summary_stats["n_vars"]
        self.n_drugs = len(drug_encoder)
        self.covars_encoder = covars_encoder
        self.split_key = split_key

        mappings = self.adata.uns['_scvi']['categorical_mappings']
        self.drugs = mappings[f'{_CE_CONSTANTS.DRUG_KEY}_scvi']['mapping']
        self.covars = {
            _CE_CONSTANTS.COVARS_KEYS[i]: list(mappings[f'{_CE_CONSTANTS.COVARS_KEYS[i]}_scvi']['mapping']) \
            for i in range(len(self.covars_encoder.keys()))
        }

        self.module = CPAModule(
            n_genes=self.n_genes,
            n_drugs=self.n_drugs,
            covars_encoder=self.covars_encoder,
            n_latent=n_latent,
            loss_ae=loss_ae,
            doser_type=doser_type,
            **hyper_params,
        )

        self.drug_encoder = drug_encoder

        val_idx, train_idx, test_idx = None, None, None
        if split_key is not None:
            train_idx = np.where(adata.obs.loc[:, split_key] == "train")[0]
            val_idx = np.where(adata.obs.loc[:, split_key] == "test")[0]
            test_idx = np.where(adata.obs.loc[:, split_key] == "ood")[0]

        self.val_idx = val_idx
        self.train_idx = train_idx
        self.test_idx = test_idx

        self._model_summary_string = f"Compositional Perturbation Autoencoder"

        self.init_params_ = self._get_init_params(locals())

        self.epoch_history = None

    @staticmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            adata: AnnData,
            batch_key: Optional[str] = None,
            labels_key: Optional[str] = None,
            layer: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            copy: bool = False,
    ) -> Optional[AnnData]:
        """
        """

        return setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            copy=copy,
        )

    def train(
            self,
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            plan_kwargs: Optional[dict] = None,
            hyperopt: bool = False,
            save_path: Optional[str] = None,
            **trainer_kwargs,
    ):
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        manual_splitting = (
                (self.val_idx is not None)
                and (self.train_idx is not None)
                and (self.test_idx is not None)
        )
        # if manual_splitting:
        data_splitter = ManualDataSplitter(
            self.adata,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            test_idx=self.test_idx,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        # else:
        # data_splitter = DataSplitter(
        #     self.adata,
        #     train_size=train_size,
        #     validation_size=validation_size,
        #     batch_size=batch_size,
        #     use_gpu=use_gpu,
        # )
        self.training_plan = CPATrainingPlan(self.module, self.covars_encoder, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs.update({'weights_summary': 'top'})
        trainer_kwargs['check_val_every_n_epoch'] = trainer_kwargs.get('check_val_every_n_epoch', 20)
        trainer_kwargs['callbacks'] = []

        if hyperopt:
            from pytorch_lightning.loggers import TensorBoardLogger
            from ray.tune.integration.pytorch_lightning import TuneReportCallback
            from ray import tune

            hyperopt_callback = TuneReportCallback({"cpa_metric": "cpa_metric",
                                                    'val_reg_mean': 'val_reg_mean',
                                                    'val_reg_var': 'val_reg_var',
                                                    'val_disent_basal_drugs': 'val_disent_basal_drugs'},
                                                   on="validation_end")
            tensorboard_logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),

            trainer_kwargs.update({'callbacks': [hyperopt_callback], 'logger': tensorboard_logger})

        else:
            if save_path is not None:
                os.makedirs(os.path.join(save_path, 'checkpoints/'), exist_ok=True)
                checkpoint = SaveBestState(monitor='cpa_metric', mode='max', period=20, verbose=True)
                trainer_kwargs['callbacks'].append(checkpoint)

        runner = TrainRunner(
            self,
            training_plan=self.training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping_monitor="cpa_metric",
            early_stopping_mode='max',
            checkpoint_callback=True,
            **trainer_kwargs,
        )
        runner()

        self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)
        if save_path is not None:
            self.save(save_path, overwrite=True)

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = 32,
    ) -> np.ndarray:
        """Returns the basal latent variable

        Parameters
        ----------
        adata : Optional[AnnData], optional
            [description], by default None
        indices : Optional[Sequence[int]], optional
            Optional indices, by default None
        batch_size : Optional[int], optional
            Batch size to use, by default None
        """

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        latent_basal = []
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            latent_basal += [outputs["latent_basal"].cpu().numpy()]
            latent += [outputs['latent'].cpu().numpy()]

        latent_basal_adata = AnnData(X=np.concatenate(latent_basal, axis=0), obs=adata.obs.copy())
        latent_basal_adata.obs_names = adata.obs_names

        latent_adata = AnnData(X=np.concatenate(latent, axis=0), obs=adata.obs.copy())
        latent_adata.obs_names = adata.obs_names

        return latent_basal_adata, latent_adata

    @torch.no_grad()
    def predict(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = 32,
    ):
        """Counterfactual-friendly gene expression prediction
        # TODO: See if another signature makes more sense for better usability

        To produce counterfactuals,

        Returns
        -------
        Tuple
            Gene expression means and standard variations
        """
        assert self.module.loss_ae in ["gauss", 'mse']
        self.module.eval()

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        mus = []
        stds = []
        for tensors in scdl:
            _mus, _stds = self.module.get_expression(tensors)
            mus.append(_mus.detach().cpu().numpy())
            stds.append(_stds.detach().cpu().numpy())

        pred_adata_mean = AnnData(X=np.concatenate(mus, axis=0), obs=adata.obs.copy())
        pred_adata_var = AnnData(X=np.concatenate(stds, axis=0), obs=adata.obs.copy())

        pred_adata_mean.obs_names = adata.obs_names
        pred_adata_var.obs_names = adata.obs_names

        return pred_adata_mean, pred_adata_var

    @torch.no_grad()
    def get_drug_embeddings(self, doses=1.0, drug: Optional[str] = None):
        """Computes all drug embeddings

        Parameters
        ----------
        doses : float, or torch.Tensor
            Drug dose, by default 1.0
        drug: str, optional
            Drug name if single drug embedding is desired

        """
        if isinstance(doses, float):
            if drug is None:
                treatments = doses * torch.eye(self.n_drugs, device=self.device)
            else:
                treatments = doses * F.one_hot(torch.LongTensor([self.drug_encoder[drug]]).to(self.device),
                                               self.n_drugs)
        else:
            treatments = doses

        embeds = self.module.drug_network(treatments).detach().cpu().numpy()

        return embeds

    @torch.no_grad()
    def get_covar_embeddings(self, covariate: str, covariate_value: str = None):
        """Computes Covariate embeddings

        Parameters
        ----------
        covariate : str
            covariate to be computed
        covariate_value: str, Optional
            Covariate specific value for embedding computation

        """
        if covariate_value is None:
            covar_ids = torch.arange(len(self.covars_encoder[covariate]), device=self.device)
        else:
            covar_ids = torch.LongTensor([self.covars_encoder[covariate][covariate_value]]).to(self.device)
        embeddings = self.module.covars_embedding[covariate](covar_ids).detach().cpu().numpy()

        return embeddings

    def save(self, dir_path: str, overwrite: bool = False, save_anndata: bool = False, **anndata_write_kwargs):
        os.makedirs(dir_path, exist_ok=True)

        self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)
        self.epoch_history.to_csv(os.path.join(dir_path, 'history.csv'), index=False)

        return super().save(dir_path, overwrite, save_anndata, **anndata_write_kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: Optional[AnnData] = None, use_gpu: Optional[Union[str, int, bool]] = None):
        model = super().load(dir_path, adata, use_gpu)

        try:
            model.epoch_history = pd.read_csv(os.path.join(dir_path, 'history.csv'))
        except:
            print('WARNING: The history was not found. Maybe the model has not been trained on any dataset.')

        return model
