from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import DTypeLike

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def to_numpy(x: torch.Tensor, numpy_dtype: Optional[DTypeLike]=None) -> np.ndarray:
    x_np: np.ndarray = x.cpu().detach().clone().numpy()
    if numpy_dtype is not None:
        x_np = x_np.astype(numpy_dtype)
    return x_np

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def from_numpy(x: np.ndarray, device: Optional[torch.device]=None, torch_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    x_torch: torch.Tensor = torch.from_numpy(x)
    if device is not None:
        x_torch = x_torch.to(device)
    if torch_dtype is not None:
        x_torch = x_torch.to(torch_dtype)
    return x_torch

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def fix_length(x: torch.Tensor, length: int, dim=-1) -> torch.Tensor:
    """Fix the length of a tensor along a specified axis by truncating or padding with zeros.
    Args:
        x (torch.Tensor): Input tensor.
        length (int): Desired length along the specified axis.
        axis (int): Axis along which to fix the length. Default is -1 (last axis).  
    Returns:
        torch.Tensor: Tensor with the specified length along the given axis.
    """

    dim = dim if dim >= 0 else x.dim() + dim
    current_length = x.shape[dim]
    
    if current_length > length:
        return x.narrow(dim, 0, length)
    elif current_length < length:
        pad_size = [0] * (2 * x.dim())
        pad_size[2 * (x.dim() - 1 - dim) + 1] = length - current_length
        return F.pad(x, pad_size)
    else:
        return x