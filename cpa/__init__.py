import warnings

warnings.simplefilter('ignore')

from ._model import CPA
from ._module import CPAModule
from . import _plotting as pl
from ._api import ComPertAPI
from ._tuner import run_autotune

from importlib.metadata import version

package_name = "cpa-tools"
__version__ = version(package_name)

__all__ = [
    "CPA",
    "CPAModule",
    "ComPertAPI",
    "pl",
    "run_autotune"
]
