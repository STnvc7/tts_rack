#Refference: https://github.com/jik876/hifi-gan/blob/master/models.py
from typing import Literal, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d
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
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class MultiPeriodDiscriminator(Discriminator):
    def __init__(
        self,
        period=[2,3,5,7,11]
    ):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p) for p in period
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
            y_d_t, fmap_t = d(target)
            y_d_p, fmap_p = d(pred)
            y_d_ts.append(y_d_t)
            fmap_ts.append(fmap_t)
            y_d_ps.append(y_d_p)
            fmap_ps.append(fmap_p)

        output = DiscriminatorOutput(
            target=y_d_ts,
            pred=y_d_ps,
            fmap_target=fmap_ts,
            fmap_pred=fmap_ps
        )

        return output