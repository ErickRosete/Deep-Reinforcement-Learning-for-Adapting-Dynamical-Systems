from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

def get_cwd():
    if HydraConfig.initialized():
        cwd = Path(get_original_cwd())
    else:
        cwd = Path.cwd()   
    return cwd


def add_cwd(path):
    return str((get_cwd() / path).resolve())