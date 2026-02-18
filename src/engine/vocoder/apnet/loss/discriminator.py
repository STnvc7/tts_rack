from typing import Dict

from interface.model import DiscriminatorOutput
from interface.loss import LossOutput, DiscriminatorLoss
from engine._common.loss.adversarial import least_square_discriminator_loss, hinge_discriminator_loss

class APNetDiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, lambda_disc: Dict[str, float]= {"mpd": 1, "mrd": 0.1}, adv_loss: str="least_square"):
        super().__init__()
        self.lambda_disc = lambda_disc
        if adv_loss == "least_square":
            self.adv_loss = least_square_discriminator_loss
        elif adv_loss == "hinge":
            self.adv_loss = hinge_discriminator_loss
        
    def forward(self, discriminator_outputs: Dict[str, DiscriminatorOutput]) -> LossOutput:
        loss = None
        for key, out in discriminator_outputs.items():
            _loss = self.adv_loss(out.target, out.pred) * self.lambda_disc[key]
            if loss is None:
                loss = _loss
            else:
                loss += _loss
        
        assert loss is not None
        output = LossOutput(total_loss=loss, loss_components=None)
        return output