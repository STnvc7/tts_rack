import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseGenerator(nn.Module):
    def __init__(self, distribution, n_components):
        super().__init__()
        self.distribution = distribution
        self.n_components = n_components
        self.weights = nn.Parameter(torch.randn(1, n_components,1))
        self.mu = nn.Parameter(torch.randn(1, n_components,1))
        self.logvar = nn.Parameter(torch.randn(1,n_components,1))

    def forward(self, size, device):
        B, T = size
        if self.distribution == "mixture":
            weights = F.softmax(self.weights, dim=1)
            mu = self.mu
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn(B, self.n_components, T, device=device)
            y = torch.sum(weights * (mu + std * eps), dim=1)
            return y
        elif self.distribution == "uniform":
            return torch.rand(B, T, device=self.weights.device) * 2 - 1
        elif self.distribution == "normal":
            return torch.randn(B, T, device=self.weights.device)
        else:
            raise NotImplementedError(f"Unknown noise type: {self.distribution}")