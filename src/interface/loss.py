from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn

from interface.data import DataLoaderOutput
from interface.model import AcousticModelOutput, GeneratorOutput, DiscriminatorOutput, E2EModelOutput


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    loss_components: Optional[Dict[str, torch.Tensor]]
    
class AcousticModelLoss(nn.Module, ABC):
    def __init__(self):
        super(AcousticModelLoss, self).__init__()
    @abstractmethod
    def forward(
        self, 
        batch: DataLoaderOutput, 
        acoustic_output: AcousticModelOutput,
    ) -> LossOutput:
        raise NotImplementedError
    
class GeneratorLoss(nn.Module, ABC):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
    @abstractmethod
    def forward(
        self, 
        batch: DataLoaderOutput, 
        generator_output: GeneratorOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput],
    ) -> LossOutput:
        raise NotImplementedError
        
class DiscriminatorLoss(nn.Module, ABC):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
    @abstractmethod
    def forward(
        self, 
        discriminator_outputs: Dict[str, DiscriminatorOutput],
    ) -> LossOutput:
        raise NotImplementedError
        
class E2EModelLoss(nn.Module, ABC):
    def __init__(self):
        super(E2EModelLoss, self).__init__()
    @abstractmethod
    def forward(
        self, 
        batch: DataLoaderOutput, 
        e2e_output: E2EModelOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput],
    ) -> LossOutput:
        raise NotImplementedError
        