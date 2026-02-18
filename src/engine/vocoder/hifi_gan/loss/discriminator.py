from typing import List, Dict
import torch
import torch.nn.functional as F

from interface.model import DiscriminatorOutput
from interface.loss import LossOutput, DiscriminatorLoss
from engine._common.loss.adversarial import least_square_discriminator_loss

class HiFiGANDiscriminatorLoss(DiscriminatorLoss):
    def __init__(self):
        super().__init__()
        
    def forward(self, discriminator_outputs: Dict[str, DiscriminatorOutput]) -> LossOutput:
        loss = None
        for out in discriminator_outputs.values():
            _loss = least_square_discriminator_loss(out.target, out.pred)
            if loss is None:
                loss = _loss
            else:
                loss += _loss
        
        assert loss is not None
        output = LossOutput(total_loss=loss, loss_components=None)
        return output