from typing import List, Union
import torch

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::      
def normalize(x: torch.Tensor, mu: float, std: float) -> torch.Tensor:
    return (x - torch.tensor(mu)) / torch.tensor(std).clamp(min=1e-8)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def denormalize(x: torch.Tensor, mu: float, std: float) -> torch.Tensor:
    return x * torch.tensor(std) + torch.tensor(mu)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def create_mask_from_lengths(lengths: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    """
    Create a mask tensor from a list of lengths.
    Args:
        lengths (List[int]): List of lengths.
    Returns:
        torch.Tensor: Mask tensor.
    Examples:
        >>> create_mask_from_lengths([3, 5, 2])
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True],
                [ True,  True, False, False, False]])
    """
    if isinstance(lengths, torch.Tensor):
        max_length = lengths.max().item()
        indeces = torch.arange(max_length, device=lengths.device)
    else:
        max_length = max(lengths)
        lengths = torch.tensor(lengths, dtype=torch.long)
        indeces = torch.arange(max_length, device=lengths.device)
    
    return indeces < lengths.unsqueeze(1)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def duration_to_attention(durations: torch.Tensor) -> torch.Tensor:
    # durations: (batch, 1, text)
    # attention: (batch, text, frame)
    dur = durations.squeeze(1) 
    ends = torch.cumsum(dur, dim=1)
    starts = ends - dur
    T = ends[:, -1].max().long().item()
    device = durations.device
    t_range = torch.arange(T, device=device).view(1, 1, T) #type: ignore
    attention = (t_range >= starts.unsqueeze(2)) & (t_range < ends.unsqueeze(2))
    return attention.to(durations.dtype)
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def slice_segment_by_id(x: torch.Tensor, id: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.gather(x, dim=dim, index=id)