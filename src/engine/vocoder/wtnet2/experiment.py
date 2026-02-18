from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from dsp_board.transforms import istft, stft

from interface.feature import AcousticFeature
from interface.model import GeneratorOutput, Generator
from interface.loggable import Audio, Heatmap, Heatmap3D, Spectrogram, Sequence
from .shuffle_next import ShuffleConvNeXt
from .noise import NoiseGenerator
from .oscillator import anti_alias, osc

class WTNet2GeneratorExperiment(Generator):
    def __init__(
        self,
        n_sample,
        fft_size,
        hop_size,
        sample_rate,
        n_mels,
        channel,
        h_channel,
        n_groups,
        kernel_sizes,
        activation,
        filter_n_bands,
        noise_distribution,
        noise_n_components,
        lowpass = True,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_sample = n_sample
        self.lowpass = lowpass
        self.pre_f0 = nn.Conv1d(1, n_mels, 1)
        self.pre_encoder = nn.Conv2d(2, channel, 1)
        self.encoder = ShuffleConvNeXt(
            channel=channel,
            h_channel=h_channel,
            n_groups=n_groups,
            kernel_sizes=kernel_sizes,
            activation=activation
        )
        self.post_encoder = nn.Conv2d(channel, 3, 1)

        self.projs = nn.ModuleDict({
            "wt_mag": nn.Conv1d(n_mels, n_sample//2+1, 1, bias=False),
            "wt_phase": nn.Conv1d(n_mels, n_sample//2+1, 1, bias=False),
            "filter_mag": nn.Conv1d(n_mels, filter_n_bands, 1, bias=False),
        })
        
        self.noise_generator = NoiseGenerator(
            distribution=noise_distribution,
            n_components=noise_n_components
        )

    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        mel = input_feature["mel_spectrogram"]
        f0 = input_feature["pitch"]
        
        f0 = F.relu(f0) # Ensure f0 is non-negative
        
        # encode feature-------------
        mel = mel.unsqueeze(1)
        log_f0 = f0.clip(min=1e-8).log()
        z_f0 = self.pre_f0(log_f0).unsqueeze(1)
        z = torch.cat([mel, z_f0], dim=1)
        
        z = self.pre_encoder(z)
        z = self.encoder(z)
        z = self.post_encoder(z)

        wt_mag = self.projs["wt_mag"](z[:, 0]).exp().clip(max=100)
        wt_phase = self.projs["wt_phase"](z[:,1])
        wt_mag = anti_alias(wt_mag, f0, self.sample_rate, self.n_sample)
        S = wt_mag * (wt_phase.cos() + 1j*wt_phase.sin())
        S[:,0,:] = 0 # Remove DC component
        wt = torch.fft.irfft(S, dim=1)
        harmonic = osc(wt, f0, self.sample_rate, self.hop_size)

        mag = self.projs["filter_mag"](z[:, 2]).exp().clip(max=100)
        mag = mag.permute(0,2,1)
        mag = F.interpolate(mag, size=self.fft_size//2+1, mode="linear", align_corners=True)
        mag = mag.permute(0,2,1)
        H = torch.complex(mag, torch.zeros_like(mag))
        
        B, _, T = harmonic.shape
        noise = self.noise_generator((B, T), device=H.device)
        X = stft(noise, self.fft_size, self.hop_size)
        Y = X * H
        noise = istft(Y, self.fft_size, self.hop_size).unsqueeze(1)
        
        y = harmonic + noise
        
        loggable = {
            "wavetable": Heatmap3D(data=wt.squeeze(), label="wavetable"),
            "wavetable_spc": Heatmap(data=wt_mag.log().squeeze(), origin="lower", label="wavetable_spc"),
            "harmonic": Audio(data=harmonic.squeeze(), sample_rate=self.sample_rate),
            "harmonic_spc": Spectrogram(data=harmonic.squeeze(), fft_size=self.fft_size, hop_size=self.hop_size, label="harmonic_spc"),
            "noise": Audio(data=noise.squeeze(), sample_rate=self.sample_rate),
            "noise_spc": Spectrogram(data=noise.squeeze(), fft_size=self.fft_size, hop_size=self.hop_size, label="noise_spc"),
            "frequency_response": Heatmap(data=H.abs().clip(min=1e-8).log().squeeze(), origin="lower", label="frequency_response"),
            "f0": Sequence(data=f0.squeeze(), label="f0"),
        }
        return GeneratorOutput(pred=y, loggable_outputs=loggable)