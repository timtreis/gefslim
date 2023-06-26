from importlib.metadata import version

from .gefslim import GEF
from .utils import _get_h5_from_gef

__version__ = version("gefslim")
