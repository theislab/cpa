from typing import Optional

from anndata import AnnData
from scvi import settings
from scvi.dataloaders import DataSplitter, AnnDataLoader
from scvi.model._utils import parse_use_gpu_arg


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata: AnnData,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices

    def setup(self, stage: Optional[str] = None):
        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )

    def train_dataloader(self):
        if len(self.train_indices) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.train_indices,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.valid_indices) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            if len(self.valid_indices < 4096):
                data_loader_kwargs.update({'batch_size': len(self.valid_indices)})
            else:
                data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata,
                indices=self.valid_indices,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_indices) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.test_indices,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
