from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
from scvi.model._utils import parse_use_gpu_arg


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        accelerator, _, self.device = parse_use_gpu_arg(
            self.use_gpu, return_device=True
        )
        self.pin_memory = (
            True
            if (settings.dl_pin_memory_gpu_training and accelerator == "gpu")
            else False
        )

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            # if len(self.valid_indices < 4096):
            #     data_loader_kwargs.update({'batch_size': len(self.valid_indices)})
            # else:
            #     data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata_manager,
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
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
