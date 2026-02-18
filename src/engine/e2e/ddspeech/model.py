from typing import Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monotonic_align import maximum_path
from dsp_board.features import mel_spectrogram
from dsp_board.transforms import stft, istft

from interface.data import DataLoaderOutput
from interface.model import E2EModel, E2EModelOutput
from interface.loggable import Audio, Heatmap, Sequence, Spectrogram

from engine._common.tensor import normalize, denormalize, slice_segment_by_id, duration_to_attention, create_mask_from_lengths
from .module.transformer import TransformerBlock
from .module.rope import RoPE
from .module.cfm import CFM1d
from .module.convnext import ConvNeXtBlock1d, CouplingBlock1d
from .module.head import WavetableHead, AdditiveHead, LTVHead

class DDSpeech(E2EModel):
    def __init__(
        self,
        n_phonemes,
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        n_harmonics,
        n_samples,
        n_bands,
        n_timesteps,
        channels,
        h_channels,
        transformer_kernel_size,
        transformer_n_layers,
        transformer_n_groups,
        transformer_n_heads,
        transformer_p_dropout,
        convnext_kernel_size,
        convnext_n_layers,
        convnext_n_groups,
        periodic_head,
        aperiodic_head,
        mel_stats,
        pitch_stats,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.n_timesteps = n_timesteps
        self.mel_stats = mel_stats
        self.pitch_stats = pitch_stats
        
        self.text_emb = nn.Embedding(n_phonemes+1, channels, padding_idx=0)
        self.text_encoder = TransformerBlock(
            channels=channels,
            h_channels=h_channels,
            n_heads=transformer_n_heads,
            n_layers=transformer_n_layers,
            kernel_size=transformer_kernel_size,
            p_dropout=transformer_p_dropout,
        )
        self.proj_mu = nn.Conv1d(channels, n_mels, 1)
        
        self.duration_predictor = nn.Sequential(
            *[ConvNeXtBlock1d(
                channels=channels,
                h_channels=h_channels,
                kernel_size=convnext_kernel_size,
                n_groups=convnext_n_groups,
                scale=1./4
            ) for _ in range(4)]
            + [nn.Conv1d(channels, 1, 1)]
        )

        self.rope = RoPE(channels, partial=True, cache_length=1000)
        self.feature_encoder = nn.Sequential(*[
            ConvNeXtBlock1d(
                channels=channels,
                h_channels=h_channels,
                kernel_size=convnext_kernel_size,
                n_groups=convnext_n_groups,
                scale=1./convnext_n_layers
            ) for _ in range(convnext_n_layers)
        ])

        self.f0_cfm = CFM1d(
            n_bins=1,
            channels=channels,
            h_channels=h_channels,
            kernel_size=convnext_kernel_size,
            n_groups=convnext_n_groups,
            n_layers=8,
        )
        self.f0_emb = nn.Sequential(
            nn.Conv1d(1, channels, 1,),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1)
        )
        self.decoder = nn.Sequential(*[
            ConvNeXtBlock1d(
                channels=channels,
                h_channels=h_channels,
                kernel_size=convnext_kernel_size,
                n_groups=convnext_n_groups,
                scale=1./convnext_n_layers
            ) for _ in range(convnext_n_layers)
        ])
        
        if periodic_head == "wavetable":
            self.periodic_head = WavetableHead(channels, n_samples, sample_rate, hop_size)
        elif periodic_head == "additive":
            self.periodic_head = AdditiveHead(channels, n_samples, sample_rate, hop_size)
        else:
            raise ValueError(f"Invalid periodic head: {periodic_head}")
        
        if aperiodic_head == "ltv":
            self.aperiodic_head = LTVHead(channels, n_samples, fft_size, hop_size)
        else:
            raise ValueError(f"Invalid aperiodic head: {aperiodic_head}")
            
    def forward(self, batch: DataLoaderOutput) -> E2EModelOutput:
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        y_mel = batch.features["mel_spectrogram"]
        y_mel_norm = normalize(y_mel, self.mel_stats.mean, self.mel_stats.std)
        y_f0 = batch.features["pitch"]
        y_f0_norm = normalize(y_f0, self.pitch_stats.mean, self.pitch_stats.std)
        y_mask = batch.feature_mask.unsqueeze(1)
        
        z = self.text_emb(x).transpose(1, 2)
        z = self.text_encoder(z, x_mask)
        x_mu = self.proj_mu(z)
        
        # monotonic alignment search --------------------
        with torch.no_grad():
            const = 0.5 * math.log(2 * math.pi) * y_mel.shape[1]
            y_square = y_mel_norm.pow(2).sum(dim=1, keepdim=True)
            x_mu_square = x_mu.pow(2).sum(dim=1, keepdim=True).transpose(1, 2)
            y_x_mu = 2 * torch.matmul(x_mu.transpose(1, 2), y_mel_norm)
            log_likelihood = -0.5 * (const + y_square - y_x_mu + x_mu_square) # [b, text, frame]
            
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn_mask = attn_mask.squeeze(1) # [b, text, frame]
            attn = maximum_path(log_likelihood, attn_mask).detach()  # type: ignore
            
        # duration prediction ---------------------------
        B, _, T = z.shape
        dur_target = attn.sum(dim=-1).log1p().unsqueeze(1)
        dur_pred = self.duration_predictor(z.detach())
        dur_loss = F.mse_loss(dur_target.masked_select(x_mask), dur_pred.masked_select(x_mask))
        
        # calculate prior loss ---------------------
        x_mu = torch.matmul(x_mu, attn)
        mu_loss = F.mse_loss(x_mu.masked_select(y_mask), y_mel_norm.masked_select(y_mask))
        
        z = torch.matmul(z, attn)
        z = self.rope(z.unsqueeze(1)).squeeze(1)
        z = self.feature_encoder(z)
        
        # f0 prediction ---------------------------
        B, _, T = z.shape
        f0_x0 = torch.randn(B, 1, T, device=z.device)
        f0_loss = self.f0_cfm.compute_loss(y_f0_norm, f0_x0, z, mask=y_mask)
        f0_pred = self.f0_cfm(f0_x0, z, self.n_timesteps, y_mask)
        f0_pred = denormalize(f0_pred, self.pitch_stats.mean, self.pitch_stats.std)
        z = z + self.f0_emb(y_f0.clip(min=1e-8).log())
        z = self.decoder(z)
        
        # slice segment ---------------------------
        if batch.segment_id_feats is not None:
            B, C, _ = z.shape
            id = batch.segment_id_feats.unsqueeze(1).expand(B, C, -1)
            z = slice_segment_by_id(z, id,dim=-1)
            id = batch.segment_id_feats.unsqueeze(1).expand(B, -1, -1)
            y_f0 = slice_segment_by_id(y_f0, id,dim=-1)
            f0_pred = slice_segment_by_id(f0_pred, id,dim=-1)
        
        # wavetable part ---------------------------
        periodic, amp_p = self.periodic_head(z, y_f0)
        aperiodic, amp_ap = self.aperiodic_head(z)
        y = periodic + aperiodic
        
        return E2EModelOutput(
            pred=y,
            pred_features={
                "pitch": f0_pred,
            },
            outputs={
                "duration_loss": dur_loss,
                "f0_cfm_loss": f0_loss,
                "mel_mu_loss": mu_loss,
            },
            loggable_outputs={
                "attn": Heatmap(attn),
                "x_mu": Heatmap(x_mu),
                "periodic": Audio(periodic, self.sample_rate),
                "periodic_spc": Spectrogram(periodic),
                "aperiodic": Audio(aperiodic, self.sample_rate),
                "aperiodic_spc": Spectrogram(aperiodic)
            }
        )
        
    def inference(self, batch: DataLoaderOutput, control: Optional[Any] = None) -> E2EModelOutput:
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        
        z = self.text_emb(x).transpose(1, 2)
        z = self.text_encoder(z, x_mask)
        
        # duration prediction ---------------------------
        B, _, T = z.shape
        log_duration = self.duration_predictor(z) * x_mask
        duration = log_duration.exp().add(-1).clip(min=0).ceil()
        y_lengths = duration.squeeze(1).sum(dim=-1) #[B]
        y_mask = create_mask_from_lengths(y_lengths)
        attn = duration_to_attention(duration)
        
        z = torch.matmul(z, attn)
        z = self.rope(z.unsqueeze(1)).squeeze(1)
        z = self.feature_encoder(z)
        
        # f0 prediction ---------------------------
        B, _, T = z.shape
        f0_x0 = torch.randn(B, 1, T, device=z.device)
        f0_pred = self.f0_cfm(f0_x0, z, self.n_timesteps, y_mask)
        f0_pred = denormalize(f0_pred, self.pitch_stats.mean, self.pitch_stats.std)
        z = z + self.f0_emb(f0_pred.clip(min=1e-8).log())
        z = self.decoder(z)
        
        # slice segment ---------------------------
        if batch.segment_id_feats is not None:
            B, C, _ = z.shape
            id = batch.segment_id_feats.unsqueeze(1).expand(B, C, -1)
            z = slice_segment_by_id(z, id,dim=-1)
            id = batch.segment_id_feats.unsqueeze(1).expand(B, -1, -1)
            f0_pred = slice_segment_by_id(f0_pred, id,dim=-1)
        
        # wavetable part ---------------------------
        periodic, amp_p = self.periodic_head(z, f0_pred)
        aperiodic, amp_ap = self.aperiodic_head(z)
        y = periodic + aperiodic
        
        return E2EModelOutput(
            pred=y,
            pred_features={
                "pitch": f0_pred,
            },
        )