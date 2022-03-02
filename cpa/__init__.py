from ._model import CPA
from ._module import CPAModule
from . import _plotting as pl
from ._api import ComPertAPI

try:
    import importlib.metadata as importlib_metadata
except:
    import importlib_metadata

package_name = "cpa-tools"
__version__ = importlib_metadata.version(package_name)

# __all__ = [
#     "CPA",
#     "CPAModule",
#     "ComPertAPI",
# ]
