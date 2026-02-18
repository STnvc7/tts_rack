from typing import List, Dict
import torch
import torch.nn.functional as F

from interface.model import GeneratorOutput, DiscriminatorOutput
from interface.loss import LossOutput, GeneratorLoss
from interface.data import DataLoaderOutput

class WaveGradLoss(GeneratorLoss):
    def __init__(self,):
        super().__init__()
    
    def forward(
        self,
        batch: DataLoaderOutput,
        generator_output: GeneratorOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        assert generator_output.outputs is not None
        loss = generator_output.outputs["loss"]
        
        return LossOutput(
            total_loss=loss, 
            loss_components={"diffusion_loss": loss}
        )