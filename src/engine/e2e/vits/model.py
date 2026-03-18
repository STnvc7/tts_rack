from typing import Optional, Any

import math
import torch
from torch import nn
from torch.nn import functional as F

import monotonic_align

from interface.data import DataLoaderOutput
from interface.model import E2EModel, E2EModelOutput
from interface.loggable import Heatmap, Audio

from engine._common.tensor import slice_segment_by_id, create_mask_from_lengths, duration_to_attention
from .modules.encoder import TextEncoder, PosteriorEncoder
from .modules.flow import ResidualCouplingBlock
from .modules.duration import StochasticDurationPredictor, DurationPredictor
from .modules.generator import Generator

class VITS(E2EModel):
    def __init__(self, 
        n_vocab,
        sample_rate,
        fft_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_speakers = n_speakers
        self.use_sdp = use_sdp
        
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )
        self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(fft_size//2+1, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    
        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
    
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, batch: DataLoaderOutput) -> E2EModelOutput:
        if not self.training:
            inference_sample = self.inference(batch).pred.squeeze()
        else:
            inference_sample = torch.zeros(self.sample_rate)
        
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        y = batch.features["log_spectrogram"]
        y_mask = batch.feature_mask.unsqueeze(1)
        
        x, m_p, logs_p = self.enc_p(x, x_mask)
        if self.n_speakers > 1:
            sid = batch.speaker_id
            g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None
        
        z, m_q, logs_q = self.enc_q(y, y_mask, g=g)
        z_p = self.flow(z, y_mask, g=g)
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True).transpose(1, 2)
            neg_cent2 = torch.matmul(s_p_sq_r.transpose(1, 2), -0.5 * (z_p ** 2))
            neg_cent3 = torch.matmul((m_p * s_p_sq_r).transpose(1, 2), z_p)
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True).transpose(1, 2)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4 # (b, text, frame)

            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(1)
            attn_mask = attn_mask.squeeze(1) # (b, text, frame)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask).detach()  # type: ignore
        
        w = attn.unsqueeze(1).sum(-1)
        if self.use_sdp:
            duration_loss = self.dp(x.detach(), x_mask, w, g=g)
            duration_loss =  duration_loss / torch.sum(x_mask)
            duration_loss = duration_loss.sum()
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            duration_loss = torch.sum((logw - logw_)**2, [1,2])
            duration_loss = duration_loss.sum() / torch.sum(x_mask) # for averaging
    
        # expand prior
        m_p = torch.matmul(m_p, attn)
        logs_p = torch.matmul(logs_p, attn)
        
        # slice segment
        if batch.segment_id_feats is not None:
            B, C, _ = z.shape
            id = batch.segment_id_feats.unsqueeze(1).expand(B, C, -1)
            z = slice_segment_by_id(z, id,dim=-1)
        
        output = self.dec(z, g=g)
        
        return E2EModelOutput(
            pred=output,
            outputs={
                "z": z,
                "z_p": z_p,
                "m_p": m_p,
                "logs_p": logs_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "duration_loss": duration_loss
            },
            loggable_outputs={
                "alignment": Heatmap(attn.squeeze()),
                "inference_sample": Audio(inference_sample, sample_rate=self.sample_rate)
            }
        )
        
    @torch.inference_mode()
    def inference(self, batch: DataLoaderOutput, control: Optional[Any] = None) -> E2EModelOutput:
        if control is None:
            control = {}

        noise_scale_w = control.get("noise_scale_w", 0.8)
        length_scale = control.get("length_scale", 1.0)
        noise_scale = control.get("noise_scale", 0.667)
        
        x = batch.phoneme_id
        x_mask = batch.phoneme_id_mask.unsqueeze(1)
        x, m_p, logs_p = self.enc_p(x, x_mask)
        if self.n_speakers > 1:
            sid = batch.speaker_id
            g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None
            
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = create_mask_from_lengths(y_lengths)
        attn = duration_to_attention(w_ceil)
        
        m_p = torch.matmul(m_p, attn)
        logs_p = torch.matmul(logs_p, attn)
    
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z, g=g)
        
        return E2EModelOutput(pred=o)
