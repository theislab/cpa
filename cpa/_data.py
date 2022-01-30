from anndata import AnnData
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi import settings

from typing import Optional

class ManualDataSplitter(DataSplitter):
    """Manual train validation test splitter"""

    def __init__(
        self,
        adata: AnnData,
        val_idx,
        train_idx,
        test_idx,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata = adata
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