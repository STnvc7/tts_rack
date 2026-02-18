from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensor import fix_length
from engine._common.tensor import create_mask_from_lengths


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration, max_length=None):
        B, N, C = x.shape
        out = []
        for i in range(B):
            _out = torch.repeat_interleave(x[i], duration[i].clip(min=0), dim=0)
            out += [_out]
        mel_lengths = [o.shape[0] for o in out]
        max_length = max_length if max_length is not None else max(mel_lengths)
        out = torch.stack([fix_length(o, max_length, dim=0) for o in out])
        
        mel_mask = create_mask_from_lengths(mel_lengths)
        mel_mask = fix_length(mel_mask, max_length, dim=1).to(out.device)
        
        return out, mel_mask


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, channel, hidden_channel, kernel_size, dropout):
        super(VariancePredictor, self).__init__()
        
        self.conv_layer = nn.Sequential(
            Conv1d(channel, hidden_channel, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Dropout(dropout),
            Conv1d(hidden_channel, hidden_channel, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Dropout(dropout),
            Conv1d(hidden_channel, hidden_channel, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Dropout(dropout),
        )

        self.linear_layer = nn.Linear(hidden_channel, 1)

    def forward(self, encoder_output, mask=None):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x
