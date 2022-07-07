import logging
import os
from typing import Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from torch.nn import functional as F
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalJointObsField,
    ObsmField,
)

from anndata import AnnData
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.train._callbacks import SaveBestState
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from ._module import CPAModule
from ._utils import REGISTRY_KEYS
from ._task import CPATrainingPlan
from ._data import AnnDataSplitter

logger = logging.getLogger(__name__)
logger.propagate = False


class CPA(BaseModelClass):
    """CPA model

        Parameters
        ----------
        adata : Anndata
            Registered Annotation Dataset

        n_latent: int
            Number of latent dimensions used for drug and Autoencoder

        loss_ae: str
            Either `gauss` or `NB`. Autoencoder loss function.

        doser_type: str
            Type of doser network. Either `sigm`, `logsigm` or `mlp`.

        split_key : str, optional
            Key used to split the data between train test and validation.
            This must correspond to a observation key for the adata, composed of values
            'train', 'test', and 'ood'. By default None.

        **hyper_params:
            CPA's hyper-parameters.


        Examples
        --------
        >>> import cpa
        >>> import scanpy as sc
        >>> adata = sc.read('dataset.h5ad')
        >>> adata = cpa.CPA.setup_anndata(adata,
                                          drug_key='condition',
                                          dose_key='dose_val',
                                          categorical_covariate_keys=['cell_type'],
                                          control_key='control'
                                          )
        >>> hyperparams = {'autoencoder_depth': 3, 'autoencoder_width': 256}
        >>> model = cpa.CPA(adata,
                            n_latent=256,
                            loss_ae='gauss',
                            doser_type='logsigm',
                            split_key='split',
                            )
    """
    covars_encoder: dict = None
    drug_encoder: dict = None

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 128,
            loss_ae: str = 'gauss',
            doser_type: str = 'logsigm',
            split_key: str = None,
            train_split: str = 'train',
            valid_split: str = 'test',
            test_split: str = 'ood',
            **hyper_params,
    ):
        super().__init__(adata)
        self.drug_encoder = CPA.drug_encoder
        self.covars_encoder = CPA.covars_encoder

        self.n_genes = adata.n_vars
        self.n_drugs = len(self.drug_encoder)
        self.split_key = split_key

        self.drugs = list(self.drug_encoder.keys())
        self.covars = {
            covar: list(self.covars_encoder[covar].keys()) for covar in self.covars_encoder.keys()
        }

        self.module = CPAModule(
            n_genes=self.n_genes,
            n_drugs=self.n_drugs,
            covars_encoder=self.covars_encoder,
            n_latent=n_latent,
            loss_ae=loss_ae,
            doser_type=doser_type,
            **hyper_params,
        ).float()

        train_indices, valid_indices, test_indices = None, None, None
        if split_key is not None:
            train_indices = np.where(adata.obs.loc[:, split_key] == train_split)[0]
            valid_indices = np.where(adata.obs.loc[:, split_key] == valid_split)[0]
            test_indices = np.where(adata.obs.loc[:, split_key] == test_split)[0]

        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices

        self._model_summary_string = f"Compositional Perturbation Autoencoder"

        self.init_params_ = self._get_init_params(locals())

        self.epoch_history = None

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            drug_key: str,
            dose_key: str,
            control_key: str,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Annotation Data setup function

        Parameters
        ----------
        adata

        drug_key

        dose_key

        control_key

        categorical_covariate_keys

        continuous_covariate_keys

        """
        REGISTRY_KEYS.DRUG_KEY = drug_key
        REGISTRY_KEYS.COVARS_KEYS = categorical_covariate_keys
        REGISTRY_KEYS.DOSE_KEY = dose_key
        REGISTRY_KEYS.CONTROL_KEY = control_key

        drugs = adata.obs[drug_key]
        dosages = adata.obs[dose_key].astype(str)

        # get unique drugs
        drugs_names_unique = set()
        for d in np.unique(drugs):
            [drugs_names_unique.add(i) for i in d.split("+")]
        drugs_names_unique = np.array(list(drugs_names_unique))

        drugs_obsm = np.zeros((adata.n_obs, len(drugs_names_unique)))
        for i in tqdm(range(adata.n_obs)):
            cell_drugs = np.isin(drugs_names_unique, drugs[i].split('+'))
            cell_doses = np.array(dosages[i].split("+")).astype(np.float32)
            drugs_obsm[i, cell_drugs] = cell_doses

        adata.obsm['drugs_doses'] = np.array(drugs_obsm)

        drug_encoder = {drug: i for i, drug in
                        enumerate(drugs_names_unique)}

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = \
            [
                LayerField(registry_key=REGISTRY_KEYS.X_KEY, layer=None, is_count_data=False),
                NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
                ObsmField('drugs_doses', 'drugs_doses', is_count_data=False, correct_data_format=True)
            ] + [CategoricalObsField(registry_key=covar, obs_key=covar) for covar in categorical_covariate_keys]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        covars_encoder = {}
        for covar in categorical_covariate_keys:
            covars_encoder[covar] = {c: i for i, c in enumerate(
                adata_manager.registry['field_registries'][covar]['state_registry']['categorical_mapping'])}

        CPA.covars_encoder = covars_encoder
        CPA.drug_encoder = drug_encoder

        # adata.uns['_scvi']['others'] = {
        #     'drug_encoder': CPA.drug_encoder,
        #     'covars_encoder': CPA.covars_encoder,
        # }

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
        """
        Trains CPA on the given dataset

        Parameters
        ----------
        max_epochs: int
            Maximum number of epochs for training
        use_gpu: bool
            Whether to use GPU if available
        train_size: float
            Fraction of training data in the case of randomly splitting dataset to train/valdiation
                if `split_key` is not set in model's constructor
        validation_size: float
            Fraction of validation data in the case of randomly splitting dataset to train/valdiation
                if `split_key` is not set in model's constructor
        batch_size: int
            Size of mini-batches for training
        early_stopping: bool
            If `True`, EarlyStopping will be used during training on validation dataset
        plan_kwargs: dict
            `CPATrainingPlan` parameters
        save_path: str
            Path to save the model after the end of training
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        manual_splitting = (
                (self.valid_indices is not None)
                and (self.train_indices is not None)
                and (self.test_indices is not None)
        )
        if manual_splitting:
            data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )

        self.training_plan = CPATrainingPlan(self.module, self.covars_encoder, **plan_kwargs)
        trainer_kwargs["early_stopping"] = False
        trainer_kwargs.update({'weights_summary': 'top'})
        trainer_kwargs['check_val_every_n_epoch'] = trainer_kwargs.get('check_val_every_n_epoch', 20)

        es_callback = EarlyStopping(monitor='cpa_metric',
                                    patience=trainer_kwargs['early_stopping_patience'],
                                    check_on_train_epoch_end=False,
                                    verbose=False,
                                    mode='max',
                                    )

        if 'callbacks' in trainer_kwargs.keys() and isinstance(trainer_kwargs.get('callbacks'), list):
            trainer_kwargs['callbacks'] += [es_callback]
        else:
            trainer_kwargs['callbacks'] = [es_callback]

        if save_path is None:
            save_path = './'

        checkpoint = SaveBestState(monitor='cpa_metric', mode='max', period=20, verbose=False)
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

        # adata = self.adata if adata is None else adata
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

        pred_adata_mean.var_names = adata.var_names
        pred_adata_var.var_names = adata.var_names

        return pred_adata_mean, pred_adata_var

    @torch.no_grad()
    def get_drug_embeddings(self, doses=1.0, drug: Optional[str] = None):
        """Computes all drug drug

        Parameters
        ----------
        doses : float, or torch.Tensor
            Drug dose, by default 1.0
        drug: str, optional
            Drug name if single drug embedding is desired

        """
        self.module.eval()
        if isinstance(doses, float):
            if drug is None:
                treatments = doses * torch.eye(self.n_drugs, device=self.device)
            else:
                treatments = doses * F.one_hot(torch.LongTensor([self.drug_encoder[drug]]).to(self.device),
                                               self.n_drugs)
        elif isinstance(doses, np.ndarray):
            treatments = torch.tensor(doses).to(self.device).float()
        else:
            treatments = doses

        embeds = self.module.drug_network(treatments).detach().cpu().numpy()

        return embeds

    @torch.no_grad()
    def get_covar_embeddings(self, covariate: str, covariate_value: str = None):
        """Computes Covariate drug

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

        if isinstance(self.training_plan.epoch_history, dict):
            self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)
            self.epoch_history.to_csv(os.path.join(dir_path, 'history.csv'), index=False)

        return super().save(dir_path=dir_path, overwrite=overwrite, save_anndata=save_anndata, **anndata_write_kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: Optional[AnnData] = None, use_gpu: Optional[Union[str, int, bool]] = None):
        model = super().load(dir_path, adata, use_gpu)
        CPA.covars_encoder = model.covars_encoder
        CPA.drug_encoder = model.drug_encoder

        try:
            model.epoch_history = pd.read_csv(os.path.join(dir_path, 'history.csv'))
        except:
            print('WARNING: The history was not found.')

        return model
