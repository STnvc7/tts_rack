#Refference: https://github.com/jik876/hifi-gan/blob/master/models.py
from typing import Literal, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import AvgPool1d, Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import spectral_norm

from interface.model import Discriminator, GeneratorOutput, DiscriminatorOutput, E2EModelOutput

LRELU_SLOPE = 0.1

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class MultiScaleDiscriminator(Discriminator):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput],
        mode: Literal["generator", "discriminator"]
    ) -> DiscriminatorOutput:
        pred = generator_output.pred
        if mode == "discriminator":
            pred = pred.detach()
            
        y_d_ts = []
        y_d_ps = []
        fmap_ts = []
        fmap_ps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                target = self.meanpools[i-1](target)
                pred = self.meanpools[i-1](pred)
            y_d_t, fmap_t = d(target)
            y_d_p, fmap_p = d(pred)
            y_d_ts.append(y_d_t)
            fmap_ts.append(fmap_t)
            y_d_ps.append(y_d_p)
            fmap_ps.append(fmap_p)

        output= DiscriminatorOutput(
            target=y_d_ts,
            pred=y_d_ps,
            fmap_target=fmap_ts,
            fmap_pred=fmap_ps     
        )

        return output