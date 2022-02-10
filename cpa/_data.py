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
from scvi import settings

from typing import Optional, Union

from scvi.data import register_tensor_from_anndata
from scvi.data import setup_anndata
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

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


class ManualDataSplitter(pl.LightningDataModule):
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
        super().__init__()
        self.adata = adata
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.use_gpu = use_gpu
        self.data_loader_kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

        if "_scvi" not in self.adata.uns.keys():
            raise ValueError("Please run setup_anndata() on your anndata object first.")

        # data_and_attributes = None
        # data_registry = self.adata.uns["_scvi"]["data_registry"]
        # for key in data_and_attributes.keys():
        #     if key not in data_registry.keys():
        #         raise ValueError(
        #             "{} required for model but not included when setup_anndata was run".format(
        #                 key
        #             )
        #         )

        self.dataset = AnnTorchDataset(self.adata, getitem_tensors=None)

    def train_dataloader(self):
        return AnnotationDataLoader(
            self.dataset,
            indices=self.train_idx,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )
            
    
    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            data_loader_kwargs.update({'batch_size': len(self.val_idx)})
            return AnnotationDataLoader(
                self.dataset,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnotationDataLoader(
                self.dataset,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

class AnnotationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        shuffle=False,
        indices=None,
        batch_size=128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):

        self.dataset = dataset

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        # sampler = SubsetRandomSampler(list(indices))
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)