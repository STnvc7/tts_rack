from typing import Tuple, Optional, Literal, Union
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


from interface.model import Discriminator, GeneratorOutput, DiscriminatorOutput, E2EModelOutput
from dsp_board.transforms import stft

class MultiBandDiscriminator(Discriminator):
    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (2048, 1024, 512),
        hop_sizes: Tuple[int, ...] = (512, 256, 128),
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    ):
        """
        Multi-Resolution Discriminator module adapted from https://github.com/descriptinc/descript-audio-codec.
        Additionally, it allows incorporating conditional information with a learned embeddings table.

        Args:
            fft_sizes (tuple[int]): Tuple of window lengths for FFT. Defaults to (2048, 1024, 512).
            bands (tuple[tuple[float, float]]): Tuple of frequency bands (as fractions of Nyquist) for each sub-discriminator. Defaults to ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)).
        """

        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(fft_sizes[i], hop_sizes[i], bands) for i in range(len(fft_sizes))]
        )

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
        y_d_ps = []
        fmap_ts = []
        fmap_ps = []
        for i, d in enumerate(self.discriminators):
            y_d_t, fmap_t = d(target)
            y_d_p, fmap_p = d(generated)
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

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
        channels: int = 32,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = fft_size
        n_fft = fft_size // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = stft(x, fft_size=self.fft_size, hop_size=self.hop_size, window_size=self.window_size)
        x = torch.view_as_real(x)
        x = x.permute(0,3,2,1)
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor):
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap