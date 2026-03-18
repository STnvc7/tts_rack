# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log as ln


class Conv1d(nn.Conv1d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight)
    if self.bias is not None:
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x, noise_level):
    """
    Arguments:
      x:
          (shape: [N,C,T], dtype: float32)
      noise_level:
          (shape: [N], dtype: float32)

    Returns:
      noise_level:
          (shape: [N,C,T], dtype: float32)
    """
    return (x + self._build_encoding(noise_level)[:, :, None])

  def _build_encoding(self, noise_level):
    count = self.dim // 2
    step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
    encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
    encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
    return encoding


class FiLM(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.encoding = PositionalEncoding(input_size)
    self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
    self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.input_conv.weight)
    nn.init.xavier_uniform_(self.output_conv.weight)
    if self.input_conv.bias is not None:
        nn.init.zeros_(self.input_conv.bias)
    if self.output_conv.bias is not None:
        nn.init.zeros_(self.output_conv.bias)

  def forward(self, x, noise_scale):
    x = self.input_conv(x)
    x = F.leaky_relu(x, 0.2)
    x = self.encoding(x, noise_scale)
    shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
    return shift, scale


class UBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor, dilation):
    super().__init__()

    self.factor = factor
    self.block1 = Conv1d(input_size, hidden_size, 1)
    self.block2 = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
    ])
    self.block3 = nn.ModuleList([
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
        Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
    ])

  def forward(self, x, film_shift, film_scale):
    block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
    block1 = self.block1(block1)

    block2 = F.leaky_relu(x, 0.2)
    block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
    block2 = self.block2[0](block2)
    block2 = film_shift + film_scale * block2
    block2 = F.leaky_relu(block2, 0.2)
    block2 = self.block2[1](block2)

    x = block1 + block2

    block3 = film_shift + film_scale * x
    block3 = F.leaky_relu(block3, 0.2)
    block3 = self.block3[0](block3)
    block3 = film_shift + film_scale * block3
    block3 = F.leaky_relu(block3, 0.2)
    block3 = self.block3[1](block3)

    x = x + block3
    return x


class DBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor):
    super().__init__()
    self.factor = factor
    self.residual_dense = Conv1d(input_size, hidden_size, 1)
    self.conv = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
        Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
        Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
    ])

  def forward(self, x):
    size = x.shape[-1] // self.factor

    residual = self.residual_dense(x)
    residual = F.interpolate(residual, size=size)

    x = F.interpolate(x, size=size)
    for layer in self.conv:
      x = F.leaky_relu(x, 0.2)
      x = layer(x)

    return x + residual


class WaveGradGenerator(nn.Module):
  def __init__(
      self, 
      n_mels,
      upsample_rate=[4,4,4,2,2],
      upsample_channels=[(768, 512), (512, 512), (512, 256), (256, 128), (128, 128)],
      upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
      downsample_channels=[(1, 32), (32, 128), (128, 128), (128, 256), (256, 512)],
  ):
    super().__init__()
    self.downsample = nn.ModuleList([])
    for i, (in_ch, out_ch) in enumerate(downsample_channels):
        if i == 0:
            _block = Conv1d(in_ch, out_ch, 5, padding=2)
        else:
            _block = DBlock(in_ch, out_ch, upsample_rate[-i])
        self.downsample.append(_block)
    
    self.film = nn.ModuleList([])
    for (_, d_out), (_, u_out) in zip(downsample_channels, reversed(upsample_channels)):
        self.film.append(FiLM(d_out, u_out))
    
    self.first_conv = Conv1d(n_mels, upsample_channels[0][0], 5, padding=2)
    self.upsample = nn.ModuleList([])
    for i, (in_ch, out_ch) in enumerate(upsample_channels):
        _block = UBlock(in_ch, out_ch, upsample_rate[i], upsample_dilations[i])
        self.upsample.append(_block)
        
    self.last_conv = Conv1d(upsample_channels[-1][1], 1, 3, padding=1)

  def forward(self, audio, spectrogram, noise_scale):
    x = audio
    downsampled = []
    for film, layer in zip(self.film, self.downsample):
        x = layer(x)
        film_out = film(x, noise_scale)
        downsampled.append(film_out)
    
    x = self.first_conv(spectrogram)
    for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
        x = layer(x, film_shift, film_scale)
    x = self.last_conv(x)
    return x