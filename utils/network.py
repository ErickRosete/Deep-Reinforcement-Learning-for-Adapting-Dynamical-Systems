import torch
from pybulletX.utils.space_dict import SpaceDict

def transform_to_tensor(x, dtype=torch.float, grad=True):
    if isinstance(x, dict):
        tensor = {k: torch.tensor(v, dtype=dtype, device="cuda", requires_grad=grad) for k, v in x.items()}
    else:
        tensor = torch.tensor(x, dtype=dtype, device="cuda", requires_grad=grad) #B,S_D
    return tensor

def calculate_input_dim(observation_space, tact_output):
    if isinstance(observation_space, SpaceDict):
        fc_input_dim = 0
        keys = list(observation_space.keys())
        if "force" in keys:
            fc_input_dim += 2
        if "position" in keys:
            fc_input_dim += 3
        if "tactile_sensor" in keys:
            fc_input_dim += tact_output
        return fc_input_dim
    return observation_space.shape[0]

def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding)//stride + 1
    height = (h - kernel_size + 2 * padding)//stride + 1
    return width, height