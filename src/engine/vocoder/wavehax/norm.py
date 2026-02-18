# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalization modules."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NormLayer(nn.Module):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the NormLayer module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        x: Tensor,
        dim: int,
        mean: Optional[Tensor] = None,
        var: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, ...).
            dim (int): Dimensions along which statistics are calculated.
            mean (Tensor, optional): Mean tensor (default: None).
            var (Tensor, optional): Variance tensor (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized tensor and statistics.
        """
        # Calculate the mean along dimensions to be reduced
        if mean is None:
            mean = x.mean(dim, keepdim=True)

        # Centerize the input tensor
        x = x - mean

        # Calculate the variance
        if var is None:
            var = (x**2).mean(dim=dim, keepdim=True)

        # Normalize
        x = x / torch.sqrt(var + self.eps)

        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)

        return x, mean, var


class LayerNorm2d(NormLayer):
    def __init__(
        self, channels: int, eps: Optional[float] = 1e-6, affine: Optional[bool] = True
    ) -> None:
        """
        Initialize the LayerNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        self.reduced_dim = [1, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x