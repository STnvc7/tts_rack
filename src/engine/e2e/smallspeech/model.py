from typing import Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monotonic_align import maximum_path
from dsp_board.features import mel_spectrogram

from interface.data import DataLoaderOutput
from interface.model import E2EModel, E2EModelOutput
from interface.loggable import Audio, Heatmap, Spectrogram

from engine._common.tensor import slice_segment_by_id, duration_to_attention, create_mask_from_lengths
from .module.conv import ConvNeXtBlock
from .module.task import TaskEmbedding
from .module.head import WTNetHead, DDSPHead

class SmallSpeech(E2EModel):
    def __init__(
        self,
        n_phonemes,
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        n_samples,
        n_bands,
        channels,
        h_channels,
        kernel_size,
        n_layers,
        n_groups,
        adaptive_norm=True,
        act="tangma",
        head="wtnet"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.text_emb = nn.Embedding(n_phonemes+1, channels, padding_idx=0)
        self.duration_proj = nn.Conv1d(channels, 1, 1)
        self.mu_proj = nn.Conv1d(channels, n_mels, 1)
        self.f0_proj = nn.Conv1d(channels, 1, 1)
        self.mel_fn = lambda x: mel_spectrogram(x, sample_rate, fft_size, hop_size, n_mels)
        self.mel_emb = nn.Conv1d(n_mels, channels, 1)
        
        if head == "wtnet":
            Head = WTNetHead
        elif head == "ddsp":
            Head = DDSPHead
        else:
            raise ValueError(f"Unsupported head type: {head}")
        self.pre_head = Head(channels, sample_rate, fft_size, hop_size, n_samples, n_bands)
        self.head = Head(channels, sample_rate, fft_size, hop_size, n_samples, n_bands)
        
        self.task_emb = nn.ModuleDict({
            "text":  TaskEmbedding(channels, h_channels),
            "duration": TaskEmbedding(channels, h_channels),
            "feature": TaskEmbedding(channels, h_channels),
            "f0": TaskEmbedding(channels, h_channels),
            "decoder": TaskEmbedding(channels, h_channels),
        })
        
        self.block = ConvNeXtBlock(
            channels=channels,
            h_channels=h_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_groups=n_groups,
            adaptive_norm=adaptive_norm,
            act=act,
        )

    def forward(self, batch: DataLoaderOutput) -> E2EModelOutput:
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        y_mel = batch.features["mel_spectrogram"]
        y_f0 = batch.features["pitch"]
        y_mask = batch.feature_mask.unsqueeze(1)
        
        z = self.text_emb(x).transpose(1, 2)
        z_text = self.block(z, x_mask, self.task_emb["text"])
        mel_mu = self.mu_proj(z_text)
        
        # monotonic alignment search --------------------
        with torch.no_grad():
            const = 0.5 * math.log(2 * math.pi) * y_mel.shape[1]
            y_square = y_mel.pow(2).sum(dim=1, keepdim=True)
            x_mu_square = mel_mu.pow(2).sum(dim=1, keepdim=True).transpose(1, 2)
            y_x_mu = 2 * torch.matmul(mel_mu.transpose(1, 2), y_mel)
            log_likelihood = -0.5 * (const + y_square - y_x_mu + x_mu_square) # [b, text, frame]
            
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn_mask = attn_mask.squeeze(1) # [b, text, frame]
            attn = maximum_path(log_likelihood, attn_mask).detach()  # type: ignore
            
        # duration prediction ---------------------------
        z_dur = self.block(z_text, x_mask, self.task_emb["duration"])
        dur_pred = self.duration_proj(z_dur)
        dur_target = attn.sum(dim=-1).log1p().unsqueeze(1)
        dur_loss = F.mse_loss(dur_target.masked_select(x_mask), dur_pred.masked_select(x_mask))
        
        mel_mu = torch.matmul(mel_mu, attn)
        mu_loss = F.mse_loss(y_mel.masked_select(y_mask), mel_mu.masked_select(y_mask))
        
        z_feature = torch.matmul(z_text, attn)
        z_feature = self.block(z_feature, y_mask, self.task_emb["feature"])
        
        z_f0 = self.block(z_feature, y_mask, self.task_emb["f0"])
        log_f0_pred = self.f0_proj(z_f0)
        log_f0_target = y_f0.log1p()
        f0_loss = F.mse_loss(log_f0_target.masked_select(y_mask), log_f0_pred.masked_select(y_mask))
        
        # slice segment ---------------------------
        if batch.segment_id_feats is not None:
            B, C, _ = z.shape
            id = batch.segment_id_feats.unsqueeze(1)
            z_feature_clip = slice_segment_by_id(z_feature, id.expand(B, C, -1), dim=-1)
            y_f0_clip = slice_segment_by_id(y_f0, id.expand(B, 1, -1), dim=-1)
            y_mask_clip = slice_segment_by_id(y_mask, id.expand(B, 1, -1), dim=-1)
            y_mel_clip = slice_segment_by_id(y_mel, id.expand(B, self.n_mels, -1), dim=-1)
        else:
            z_feature_clip = z_feature
            y_f0_clip = y_f0
            y_mask_clip = y_mask
            y_mel_clip = y_mel
        
        pre, _, _ = self.pre_head(z_feature_clip, y_f0_clip)
        pre_mel = self.mel_fn(pre.squeeze(1))
        pre_loss = F.mse_loss(y_mel_clip.masked_select(y_mask_clip), pre_mel.masked_select(y_mask_clip))
        
        z_decoder = z_feature_clip + self.mel_emb(pre_mel.detach())
        z_decoder = self.block(z_decoder, y_mask_clip, self.task_emb["decoder"])
        y, periodic, aperiodic = self.head(z_decoder, y_f0_clip)
        
        return E2EModelOutput(
            pred=y,
            pred_features={
                "pitch": log_f0_pred.expm1(),
            },
            outputs={
                "duration_target": dur_target,
                "duration_pred": dur_pred,
                "log_f0_target": log_f0_target,
                "log_f0_pred": log_f0_pred,
                "duration_loss": dur_loss,
                "f0_loss": f0_loss,
                "mu_loss": mu_loss,
                "prehead_loss": pre_loss,
                "z_text": z_text,
                "z_feature": z_feature,
                "x_mask": x_mask,
                "y_mask": y_mask,
            },
            loggable_outputs={
                "attn": Heatmap(attn.squeeze(0)),
                "attn_pred": Heatmap(duration_to_attention(dur_pred.expm1().clip(min=0).ceil())),
                "mel_mu": Heatmap(mel_mu.squeeze()),
                "pre": Audio(pre.squeeze(), self.sample_rate),
                "pre_spc": Spectrogram(pre.squeeze()),
                "periodic": Audio(periodic.squeeze(), self.sample_rate),
                "periodic_spc": Spectrogram(periodic.squeeze()),
                "aperiodic": Audio(aperiodic.squeeze(), self.sample_rate),
                "aperiodic_spc": Spectrogram(aperiodic.squeeze())
            }
        )
        
    def inference(self, batch: DataLoaderOutput, control: Optional[Any] = None) -> E2EModelOutput:
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        z = self.text_emb(x).transpose(1, 2)
        z = self.block(z, x_mask, self.task_emb["text"])
        
        z_dur = self.block(z, x_mask, self.task_emb["duration"])
        log_duration = self.duration_proj(z_dur)
        duration = log_duration.expm1().clip(min=0).ceil()
        y_lengths = duration.squeeze(1).sum(dim=-1) #[B]
        y_mask = create_mask_from_lengths(y_lengths)
        attn = duration_to_attention(duration)
        
        z = torch.matmul(z, attn)
        z = self.block(z, y_mask, self.task_emb["feature"])
        z_f0 = self.block(z, y_mask, self.task_emb["f0"])
        f0_pred = self.f0_proj(z_f0).expm1()
        
        # slice segment ---------------------------
        if batch.segment_id_feats is not None:
            B, C, _ = z.shape
            id = batch.segment_id_feats.unsqueeze(1)
            z = slice_segment_by_id(z, id.expand(B, C, -1), dim=-1)
            f0_pred = slice_segment_by_id(f0_pred, id.expand(B, 1, -1), dim=-1)
        
        pre, _, _ = self.pre_head(z, f0_pred)
        pre_mel = self.mel_fn(pre.squeeze(1))
        z = z + self.mel_emb(pre_mel)
        z = self.block(z, y_mask, self.task_emb["decoder"])
        y, _, _ = self.head(z, f0_pred)
        
        return E2EModelOutput(
            pred=y,
            pred_features={
                "pitch": f0_pred,
            },
        )