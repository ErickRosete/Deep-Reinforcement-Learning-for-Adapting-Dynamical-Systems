import torch

def transform_to_tensor(x, dtype=torch.float, grad=True):
    if isinstance(x, dict):
        tensor = {k: torch.tensor(v, dtype=dtype, device="cuda", requires_grad=grad) for k, v in x.items()}
    else:
        tensor = torch.tensor(x, dtype=dtype, device="cuda", requires_grad=grad) #B,S_D
    return tensor

def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding)//stride + 1
    height = (h - kernel_size + 2 * padding)//stride + 1
    return width, height