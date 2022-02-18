import os
import pickle
import inspect
import pandas as pd


from typing import Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData

from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainRunner

from ._module import CPAModule, _CE_CONSTANTS
from ._task import CPATrainingPlan
from ._data import ManualDataSplitter

class CPA(BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_latent: int,
        loss_ae: str,
        doser_type: str,
        covars_to_ncovars: dict,
        split_key: str = None,
        drug_encoder = None,
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
        self.n_drugs = adata.obsm[_CE_CONSTANTS.PERTURBATIONS].shape[-1]
        self.covars_to_ncovars = covars_to_ncovars
        
        mappings = self.adata.uns['_scvi']['categorical_mappings']
        self.drugs = mappings[f'{_CE_CONSTANTS.DRUG_KEY}_scvi']['mapping']
        self.covars = {
            _CE_CONSTANTS.COVARS_KEYS[i]: list(mappings[f'{_CE_CONSTANTS.COVARS_KEYS[i]}_scvi']['mapping']) \
            for i in range(len(self.covars_to_ncovars.keys()))
        }
        
        self.module = CPAModule(
            n_genes=self.n_genes,
            n_drugs=self.n_drugs,
            covars_to_ncovars=self.covars_to_ncovars,
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
            num_workers=6,
        )
        # else:
            # data_splitter = DataSplitter(
            #     self.adata,
            #     train_size=train_size,
            #     validation_size=validation_size,
            #     batch_size=batch_size,
            #     use_gpu=use_gpu,
            # )
        self.training_plan = CPATrainingPlan(self.module, self.covars_to_ncovars, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs.update({'weights_summary': 'top'})
        trainer_kwargs['check_val_every_n_epoch'] = trainer_kwargs.get('check_val_every_n_epoch', 20)

        if hyperopt:
            from pytorch_lightning.loggers import TensorBoardLogger
            from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
            from ray import tune

            hyperopt_callback = TuneReportCallback({"cpa_metric": "cpa_metric", 
                                                    'val_reg_mean': 'val_reg_mean',
                                                    'val_reg_var': 'val_reg_var',
                                                    'val_disent_basal_drugs': 'val_disent_basal_drugs'}, 
                                                    on="validation_end")
            tensorboard_logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),

            trainer_kwargs.update({'callbacks': [hyperopt_callback], 'logger': tensorboard_logger})

        runner = TrainRunner(
            self,
            training_plan=self.training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping_monitor="cpa_metric",
            early_stopping_mode='max',
            **trainer_kwargs,
        )
        runner()
        
        self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)

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
            adata=adata, indices=indices, batch_size=batch_size, num_workers=6, shuffle=False
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
    def get_drug_embeddings(self, dose=1.0) -> AnnData:
        """Computes all drug embeddings

        Parameters
        ----------
        dose : float, optional
            Drug dose, by default 1.0

        """
        treatments = dose * torch.eye(self.n_drugs, device=self.device)
        embeds = self.module.drug_network(treatments).detach().cpu().numpy()

        drug_adata = AnnData(X=embeds, obs={'drug_name': list(self.drugs)}) 
        return drug_adata

    @torch.no_grad()
    def get_covar_embeddings(self, covariate: str):
        """Computes Covariate embeddings

        Parameters
        ----------
        covariate : str
            covariate to be computed

        """
        key = f'covar_{covariate}'
        covar_ids = torch.arange(self.covars_to_ncovars[key], device=self.device)
        embeddings = self.module.covars_embedding[key](covar_ids).detach().cpu().numpy()

        covar_adata = AnnData(X=embeddings, obs={f'{covariate}': self.covars[covariate]})
        return covar_adata

    @torch.no_grad()
    def get_disentanglement_score(self, adata: Optional[AnnData] = None, 
                                        indices: Optional[Sequence[int]] = None,
                                        batch_size: Optional[int] = 32):

        pass

    def latent_dose_response(self, drugs=None, 
                                   dose=None, 
                                   contvar_min=0, 
                                   contvar_max=1, 
                                   n_points=100):
        """
        Parameters
        ----------
        drugs : list
            List of drug names.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.
        Returns
        -------
        pd.DataFrame
        """
        if drugs is None:
            drugs = self.drugs

        if dose is None:
            dose = np.linspace(contvar_min, contvar_max, n_points)

        n_points = len(dose)

        df = pd.DataFrame(columns=[_CE_CONSTANTS.DRUG_KEY, _CE_CONSTANTS.DOSE_KEY, 'response'])

        for drug in drugs:
            d = np.where(self.drugs == drug)[0][0]
            
            this_drug = torch.Tensor(dose).to(self.device).view(-1, 1)
            
            # TODO: Do this for doser_type == 'mlp'
            response = self.module.drug_network.dosers.one_drug(this_drug.view(-1), d)
            response = list(response.detach().cpu().numpy().reshape(-1))

            df_drug = pd.DataFrame(list(zip([drug] * n_points, dose, response)), columns=df.columns)
            df = pd.concat([df, df_drug])

        return df.reset_index(drop=True)

    def latent_dose_response2D(self, perturbations, dose=None,
        contvar_min=0, contvar_max=1, n_points=100,):
        """
        Parameters
        ----------
        perturbations : list, optional (default: None)
            List of atomic drugs for which to return latent dose response.
            Currently drug combinations are not supported.
        doses : np.array (default: None)
            Doses values. If None, default values will be generated on a grid:
            n_points in range [contvar_min, contvar_max].
        contvar_min : float (default: 0)
            Minimum dose value to generate for default option.
        contvar_max : float (default: 0)
            Maximum dose value to generate for default option.
        n_points : int (default: 100)
            Number of dose points to generate for default option.
        Returns
        -------
        pd.DataFrame
        """
        # dosers work only for atomic drugs. TODO add drug combinations

        assert len(perturbations) == 2, "You should provide a list of 2 perturbations."

        if dose is None:
            dose = np.linspace(contvar_min, contvar_max, n_points)
        n_points = len(dose)

        df = pd.DataFrame(columns=perturbations + ['response'])
        response = {}

        for drug in perturbations:
            d = np.where(self.drugs == drug)[0][0]
            this_drug = torch.Tensor(dose).to(self.device).view(-1, 1)
            
            # TODO: doser_type == mlp implementation
            response[drug] = self.module.drug_network.dosers.one_drug(this_drug.view(-1),\
                d).detach().cpu().numpy().reshape(-1)

        l = 0
        for i in range(len(dose)):
            for j in range(len(dose)):
                df.loc[l] = [dose[i], dose[j], response[perturbations[0]][i] + response[perturbations[1]][j]]
                l += 1

        return df.reset_index(drop=True)
    
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
