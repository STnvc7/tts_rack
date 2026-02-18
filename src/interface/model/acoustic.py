from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from interface.loggable import Loggable
from interface.data import DataLoaderOutput
from interface.feature import AcousticFeature

@dataclass
class AcousticModelOutput:
    pred_features: Dict[AcousticFeature, torch.Tensor]
    outputs: Optional[Dict[str, torch.Tensor]] = None
    loggable_outputs: Optional[Dict[str, Loggable]] = None


class AcousticModel(nn.Module, ABC):
    def __init__(self):
        super(AcousticModel, self).__init__()
    @abstractmethod
    def forward(
        self, 
        batch: DataLoaderOutput
    ) -> AcousticModelOutput:
        raise NotImplementedError
    @abstractmethod
    def inference(
        self, 
        batch: DataLoaderOutput, 
        control: Optional[Any]=None
    ) -> AcousticModelOutput:
        raise NotImplementedError