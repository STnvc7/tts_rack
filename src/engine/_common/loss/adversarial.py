from typing import List
import torch
import torch.nn.functional as F

def feature_matching_loss(fmap_target: List[List[torch.Tensor]], fmap_generated: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = None
    for dt, dg in zip(fmap_target, fmap_generated):
        for tl, gl in zip(dt, dg):
            _loss = torch.mean(torch.abs(tl - gl))
            if loss is None:
                loss = _loss
            else:
                loss += _loss
    assert loss is not None
    return loss

def least_square_generator_loss(disc_gen_outputs: List[torch.Tensor]) -> torch.Tensor:
    loss = None
    for dg in disc_gen_outputs:
        _loss = torch.mean((1-dg)**2)
        if loss is None:
            loss = _loss
        else:
            loss += _loss
    assert loss is not None
    return loss
    
def least_square_discriminator_loss(target: List[torch.Tensor], generated: List[torch.Tensor]) -> torch.Tensor:
    loss = None
    for dt, dg in zip(target, generated):
        t_loss = torch.mean((1-dt)**2)
        g_loss = torch.mean(dg**2)
        _loss = t_loss + g_loss
        if loss is None:
            loss = _loss
        else:
            loss += _loss
    assert loss is not None
    return loss

def hinge_generator_loss(disc_gen_outputs: List[torch.Tensor]) -> torch.Tensor:
    loss = None
    for dg in disc_gen_outputs:
        _loss = torch.mean(torch.clamp(1-dg, min=0))
        if loss is None:
            loss = _loss
        else:
            loss += _loss
    assert loss is not None
    return loss
    
def hinge_discriminator_loss(target: List[torch.Tensor], generated: List[torch.Tensor]) -> torch.Tensor:
    loss = None
    for dt, dg in zip(target, generated):
        t_loss = torch.mean(torch.clamp(1-dt, min=0))
        g_loss = torch.mean(torch.clamp(1+dg, min=0))
        _loss = t_loss + g_loss
        if loss is None:
            loss = _loss
        else:
            loss += _loss
    assert loss is not None
    return loss