from typing import Optional, Dict, Any, List, Literal, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from interface.loggable import Loggable
from interface.feature import AcousticFeature
from .e2e import E2EModelOutput

@dataclass
class GeneratorOutput:
    pred: torch.Tensor
    outputs: Optional[Dict[str, torch.Tensor]] = None
    loggable_outputs: Optional[Dict[str, Loggable]] = None
    
class Generator(nn.Module, ABC):
    def __init__(self):
        super(Generator, self).__init__()
    def forward(
        self,
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        raise NotImplementedError
    def inference(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor], 
        control: Optional[Any]=None
    ) -> GeneratorOutput:
        return self(input_feature)

@dataclass
class DiscriminatorOutput:
    target: List[torch.Tensor]
    pred: List[torch.Tensor]
    fmap_target: List[List[torch.Tensor]]
    fmap_pred: List[List[torch.Tensor]]
    outputs: Optional[Dict[str, torch.Tensor]] = None

class Discriminator(nn.Module, ABC):
    def __init__(self):
        super(Discriminator, self).__init__()
    @abstractmethod
    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput], 
        mode: Literal["generator", "discriminator"]
    ) -> DiscriminatorOutput:
        raise NotImplementedError


class Discriminators(nn.Module):
    def __init__(self, discriminators: Optional[Dict[str, Discriminator]]=None):
        super(Discriminators, self).__init__()
        if discriminators is None:
            self.models = None
        else:
            self.models = nn.ModuleDict(discriminators)

    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput], 
        mode: Literal["generator", "discriminator"]
    ) -> Dict[str, DiscriminatorOutput]:
        if self.models is None:
            return {}
        
        disc_outputs = {}
        for key, d in self.models.items():
            disc_output = d(target, generator_output, mode)
            disc_outputs[key] = disc_output
        return disc_outputs