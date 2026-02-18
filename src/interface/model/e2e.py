from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from interface.loggable import Loggable
from interface.data import DataLoaderOutput
from interface.feature import AcousticFeature

@dataclass
class E2EModelOutput:
    pred: torch.Tensor
    pred_features: Optional[Dict[AcousticFeature, torch.Tensor]] = None
    outputs: Optional[Dict[str, torch.Tensor]] = None
    loggable_outputs: Optional[Dict[str, Loggable]] = None
    
class E2EModel(nn.Module, ABC):
    def __init__(self):
        super(E2EModel, self).__init__()
    @abstractmethod
    def forward(
        self, 
        batch: DataLoaderOutput
    ) -> E2EModelOutput:
        raise NotImplementedError
    @abstractmethod
    def inference(
        self, 
        batch: DataLoaderOutput, 
        control: Optional[Any]=None
    ) -> E2EModelOutput:
        raise NotImplementedError