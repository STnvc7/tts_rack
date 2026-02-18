from typing import Optional
from abc import ABC, abstractmethod
import torch
from interface.corpus import Corpus, FeatureStats


class UtteranceListMaker(ABC):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
    @abstractmethod
    def make(self):
        raise NotImplementedError

class StatsCalculator:
    def __init__(self, avoid_value: Optional[float]=None):
        self.avoid_value = avoid_value
        self.sum = torch.tensor(0.0)
        self.sum_squared = torch.tensor(0.0)
        self.count = torch.tensor(0.0)
        self.max = torch.tensor(float("-inf"))
        self.min = torch.tensor(float("inf"))

    def update(self, sequence: torch.Tensor):
        if self.avoid_value:
            sequence = sequence[sequence != self.avoid_value]
        if sequence.numel() == 0:
            return
        self.sum += torch.sum(sequence)
        self.sum_squared += torch.sum(sequence**2)
        self.count += sequence.numel()
        self.max = max(self.max, torch.max(sequence))
        self.min = min(self.min, torch.min(sequence))

    def compute(self) -> FeatureStats:
        mean = self.sum / self.count
        variance = self.sum_squared / self.count - mean**2
        mean = self.sum / self.count
        variance = self.sum_squared / self.count - mean ** 2
        std = torch.sqrt(variance)

        return FeatureStats(mean.item(), std.item(), self.max.item(), self.min.item())
