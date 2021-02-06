from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import torch

def get_cwd():
    if HydraConfig.initialized():
        cwd = Path(get_original_cwd())
    else:
        cwd = Path.cwd()   
    return cwd

def add_cwd(path):
    return str((get_cwd() / path).resolve())

def transform_to_tensor(x, dtype=torch.float, grad=True):
    if isinstance(x, dict):
        tensor = {k: torch.tensor(v, dtype=dtype, device="cuda", requires_grad=grad) for k, v in x.items()}
    else:
        tensor = torch.tensor(x, dtype=dtype, device="cuda", requires_grad=grad) #B,S_D
    return tensor