import drlfads

from pathlib                 import Path

from hydra.utils             import get_original_cwd
from hydra.core.hydra_config import HydraConfig


def get_cwd():
    if HydraConfig.initialized():
        cwd = Path(get_original_cwd())
    else:
        cwd = Path.cwd()   
    return cwd

def add_cwd(path):
    return str((get_cwd() / path).resolve())

def pkg_path(rel_path):
    """Generates a global path that is relative to
    the root of drlfads package.
    (Could be generalized for any python module)

    Args:
        rel_path (str): Relative path within drlfads package 

    Returns:
        str: Global path.
    """
    return str(Path(drlfads.__path__[0], rel_path))