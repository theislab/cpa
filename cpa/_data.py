import copy
import torch


import numpy as np
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder

from anndata import AnnData
import pytorch_lightning as pl
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._ann_dataloader import AnnDataLoader, BatchSampler
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi.dataloaders import DataSplitter
from scvi import settings

from typing import Optional, Union

from scvi.data import register_tensor_from_anndata
from scvi.data import setup_anndata
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from ._utils import _CE_CONSTANTS

def prepare_dataset(
    data_path,
    drug_key,
    dose_key,
    covars_keys,
):
    """TEMPORARY.

    Quick and dirty way to construct the dataloader for the CPA model.
    This function will be replaced once the AnnData refactor is completed within
    scvi-tools.

    Parameters
    ----------
    adata : AnnData
    drug_key : str
        Obs key for the drug names
    dose_key : str
        Obs key for drug doses
    covars_keys : list
        List of categorical covariates
    """
    _CE_CONSTANTS.DRUG_KEY = drug_key
    _CE_CONSTANTS.COVARS_KEYS = covars_keys
    _CE_CONSTANTS.DOSE_KEY = dose_key

    adata = sc.read(data_path)

    setup_anndata(adata)

    drugs = adata.obs[drug_key]

    # get unique drugs
    drugs_names_unique = set()
    for d in drugs:
        [drugs_names_unique.add(i) for i in d.split("+")]
    drugs_names_unique = np.array(list(drugs_names_unique))
    drug_encoder = OneHotEncoder(sparse=False)
    drug_encoder.fit(drugs_names_unique.reshape(-1, 1))

    drugs_doses = []
    for i, comb in enumerate(drugs):
        drugs_combos = drug_encoder.transform(
            np.array(comb.split("+")).reshape(-1, 1))
        dose_combos = str(adata.obs[dose_key].values[i]).split("+")
        for j, d in enumerate(dose_combos):
            if j == 0:
                drug_ohe = float(d) * drugs_combos[j]
            else:
                drug_ohe += float(d) * drugs_combos[j]
        drugs_doses.append(drug_ohe)
    
    adata.obsm['drugs_doses'] = np.array(drugs_doses)
    
    register_tensor_from_anndata(adata, "drugs_doses", "obsm", "drugs_doses")
    register_tensor_from_anndata(adata, "drug_name", "obs", drug_key, is_categorical=True)
    covars_to_ncovars = dict()
    for covar in covars_keys:
        new_covar_key = f"covar_{covar}"
        register_tensor_from_anndata(
            adata, new_covar_key, "obs", covar, is_categorical=True
        )
        covars_to_ncovars[new_covar_key] = len(adata.obs[covar].unique())
    
    return adata, drug_encoder, covars_to_ncovars

class ManualDataSplitter(DataSplitter):
    """Manual train validation test splitter"""

    def __init__(
        self,
        adata: AnnData,
        train_idx,
        val_idx,
        test_idx,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(adata)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.val_idx = val_idx
        self.train_idx = train_idx
        self.test_idx = test_idx

    def setup(self, stage: Optional[str] = None):
        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            if len(self.val_idx < 5000):
                data_loader_kwargs.update({'batch_size': len(self.val_idx)})
            else:
                data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass