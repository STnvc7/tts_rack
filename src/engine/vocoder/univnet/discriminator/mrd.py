from typing import Literal, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import spectral_norm
from dsp_board.features import spectrogram

from interface.model import Discriminator, GeneratorOutput, DiscriminatorOutput, E2EModelOutput

LRELU_SLOPE = 0.1

class SpecDiscriminator(nn.Module):

    def __init__(
        self, 
        fft_size=1024, 
        shift_size=120, 
        win_length=600, 
        use_spectral_norm=False
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spc_fn = lambda x: spectrogram(
            x, 
            fft_size=self.fft_size, 
            hop_size=self.shift_size, 
            window_size=self.win_length, 
            log=True,
        )
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        # with torch.no_grad():
        y = y.squeeze(1)
        y = self.spc_fn(y)
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResolutionDiscriminator(Discriminator):

    def __init__(self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        window_sizes=[600, 1200, 240],
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_size, hop_size, window_size)
            for fft_size, hop_size, window_size in zip(fft_sizes, hop_sizes, window_sizes)
        ])

    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput],
        mode: Literal["generator", "discriminator"]
    ) -> DiscriminatorOutput:
        generated = generator_output.pred
        if mode == "discriminator":
            generated = generated.detach()

        y_d_ts = []
        y_d_gs = []
        fmap_ts = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_t, fmap_t = d(target)
            y_d_g, fmap_g = d(generated)
            y_d_ts.append(y_d_t)
            fmap_ts.append(fmap_t)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        output = DiscriminatorOutput(
            target=y_d_ts,
            pred=y_d_gs,
            fmap_target=fmap_ts,
            fmap_pred=fmap_gs,
        )

        return output
