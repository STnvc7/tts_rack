from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from interface.model import AcousticModel, AcousticModelOutput
from interface.corpus import FeatureStats
from interface.data import DataLoaderOutput
from interface.loggable import Duration, Heatmap

from .transformer import Decoder, Encoder, PostNet
from .variance_adaptor import LengthRegulator, VariancePredictor
from engine._common.tensor import normalize, denormalize

class FastSpeech2(AcousticModel):
    pitch_bins: torch.Tensor
    energy_bins: torch.Tensor
    def __init__(
        self,
        n_mels: int,
        n_speakers: int,
        max_length: int,
        n_phonemes: int,
        encoder_channel: int,
        encoder_hidden_channel: int,
        encoder_n_layers: int,
        encoder_n_heads: int,
        encoder_kernel_size: int,
        encoder_dropout: float,
        decoder_channel: int,
        decoder_hidden_channel: int,
        decoder_n_layers: int,
        decoder_n_heads: int,
        decoder_kernel_size: int,
        decoder_dropout: float,
        variance_adaptor_hidden_channel: int,
        variance_adaptor_kernel_size: int,
        variance_adaptor_dropout: float,
        feature_emb_quantize_bin: int,
        pitch_stats: FeatureStats,
        energy_stats: FeatureStats,
    ):
        super(FastSpeech2, self).__init__()

        self.quantize_bin = feature_emb_quantize_bin
        self.pitch_mean = pitch_stats.mean
        self.pitch_std = pitch_stats.std
        self.pitch_min = pitch_stats.min
        self.pitch_max = pitch_stats.max
        self.energy_mean = energy_stats.mean
        self.energy_std = energy_stats.std
        self.energy_min = energy_stats.min
        self.energy_max = energy_stats.max

        self.speaker_emb = nn.Embedding(n_speakers, encoder_channel)

        self.encoder = Encoder(
            max_length=max_length,
            n_phonemes=n_phonemes,
            channel=encoder_channel,
            hidden_channel=encoder_hidden_channel,
            n_layers=encoder_n_layers,
            n_heads=encoder_n_heads,
            kernel_size=encoder_kernel_size,
            dropout=encoder_dropout,
        )

        self.decoder = Decoder(
            max_length=max_length,
            channel=decoder_channel,
            hidden_channel=decoder_hidden_channel,
            n_layers=decoder_n_layers,
            n_heads=decoder_n_heads,
            kernel_size=decoder_kernel_size,
            dropout=decoder_dropout,
        )

        self.duration_predictor = VariancePredictor(
            channel=encoder_channel,
            hidden_channel=variance_adaptor_hidden_channel,
            kernel_size=variance_adaptor_kernel_size,
            dropout=variance_adaptor_dropout,
        )
        self.length_regulator = LengthRegulator()
        
        self.pitch_predictor = VariancePredictor(
            channel=encoder_channel,
            hidden_channel=variance_adaptor_hidden_channel,
            kernel_size=variance_adaptor_kernel_size,
            dropout=variance_adaptor_dropout,
        )
        self.register_buffer("pitch_bins", torch.linspace(self.pitch_min, self.pitch_max, self.quantize_bin-1))
        self.pitch_emb = nn.Embedding(feature_emb_quantize_bin, encoder_channel)

        self.energy_predictor = VariancePredictor(
            channel=encoder_channel,
            hidden_channel=variance_adaptor_hidden_channel,
            kernel_size=variance_adaptor_kernel_size,
            dropout=variance_adaptor_dropout,
        )
        self.register_buffer("energy_bins", torch.linspace(self.energy_min, self.energy_max, self.quantize_bin-1))
        self.energy_emb = nn.Embedding(feature_emb_quantize_bin, encoder_channel)

        self.mel_linear = nn.Linear(decoder_channel, n_mels)
        self.postnet = PostNet()

    def forward(self, batch: DataLoaderOutput) -> AcousticModelOutput:
        output = self.encoder(batch.phoneme_id, ~batch.phoneme_id_mask)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(batch.speaker_id)
        
        log_duration_pred = self.duration_predictor(output, ~batch.phoneme_id_mask)
        duration_pred = torch.clamp(torch.round(torch.exp(log_duration_pred) - 1), min=0).int()
        assert batch.duration is not None, "Duration should be provided"
        duration_round = batch.duration
        output, _ = self.length_regulator(output, duration_round, max_length=None)
    
        pitch = batch.features["pitch"].squeeze(1)
        energy = batch.features["mel_energy"].squeeze(1)
        feature_mask = batch.feature_mask
            
        # pitch prediction ----------------------
        pitch_pred_norm = self.pitch_predictor(output, ~feature_mask)
        pitch_pred_linear = denormalize(pitch_pred_norm, self.pitch_mean, self.pitch_std)
        
        # pitch embedding ----------------------
        pitch_quantized = torch.bucketize(pitch, self.pitch_bins)
        pitch_emb = self.pitch_emb(pitch_quantized)
        output = output + pitch_emb
        
        # energy prediction ----------------------
        energy_pred_norm = self.energy_predictor(output, ~feature_mask)
        energy_pred_linear = denormalize(energy_pred_norm, self.energy_mean, self.energy_std)
        
        # energy embedding ----------------------
        energy_quantized = torch.bucketize(energy, self.energy_bins)
        energy_emb = self.energy_emb(energy_quantized)
        output = output + energy_emb

        decoder_output = self.decoder(output, ~feature_mask)
        output = self.mel_linear(decoder_output)
        postnet_output = self.postnet(output) + output

        return AcousticModelOutput(
            pred_features={
                "mel_spectrogram": postnet_output.permute(0, 2, 1),
                "pitch": pitch_pred_linear.unsqueeze(1),
                "mel_energy": energy_pred_linear.unsqueeze(1),
            },
            outputs={
                "decoder_output": decoder_output,
                "postnet_output": postnet_output,
                "output": output,
                "log_duration_pred": log_duration_pred,
                "pitch_raw_norm": normalize(pitch, self.pitch_mean, self.pitch_std),
                "pitch_pred_norm": pitch_pred_norm,
                "energy_raw_norm": normalize(energy, self.energy_mean, self.energy_std),
                "energy_pred_norm": energy_pred_norm,
            },
            loggable_outputs={
                "output": Heatmap(output.permute(0,2,1).squeeze(0), label="output_before_postnet", origin="lower"),
                "duration_ground_truth": Duration(
                    batch.phoneme[0].split(), 
                    batch.duration.squeeze(0),
                    batch.features["mel_spectrogram"].squeeze(0),
                ),
                "duration_pred": Duration(
                    batch.phoneme[0].split(), 
                    duration_pred.squeeze(0),
                    batch.features["mel_spectrogram"].squeeze(0),          
                ),
            }
        )

    def inference(self, batch: DataLoaderOutput, control: Optional[Any]=None) -> AcousticModelOutput:
        if control is None:
            control = {}
        duration_alpha = control.get("duration_alpha", 1.0)
        pitch_mean_alpha = control.get("pitch_mean_alpha", 1.0)
        pitch_std_alpha = control.get("pitch_std_alpha", 1.0)
        energy_mean_alpha = control.get("energy_mean_alpha", 1.0)
        energy_std_alpha = control.get("energy_std_alpha", 1.0)
        
        output = self.encoder(batch.phoneme_id, ~batch.phoneme_id_mask)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(batch.speaker_id)
        
        log_duration_pred = self.duration_predictor(output, ~batch.phoneme_id_mask)
        duration_pred = torch.clamp(torch.round(torch.exp(log_duration_pred) - 1), min=0).int()
        output, feature_mask = self.length_regulator(output, duration_pred, max_length=None)

        # pitch prediction ----------------------
        pitch_pred = self.pitch_predictor(output, ~feature_mask)
        pitch_pred = denormalize(pitch_pred, self.pitch_mean * pitch_mean_alpha, self.pitch_std * pitch_std_alpha)
        threshold = (self.pitch_min-self.pitch_mean)*pitch_std_alpha + (self.pitch_mean*pitch_mean_alpha)
        pitch_pred = F.threshold(pitch_pred, threshold, 0)
        
        # pitch embedding ----------------------
        pitch_quantized = torch.bucketize(pitch_pred, self.pitch_bins)
        pitch_emb = self.pitch_emb(pitch_quantized)
        output = output + pitch_emb
        
        # energy prediction ----------------------
        energy_pred = self.energy_predictor(output, ~feature_mask)
        energy_pred = denormalize(energy_pred, self.energy_mean * energy_mean_alpha, self.energy_std * energy_std_alpha)
        threshold = (self.energy_min-self.energy_mean)*energy_std_alpha + (self.energy_mean*energy_mean_alpha)
        energy_pred = F.threshold(energy_pred, threshold, 0)

        # energy embedding ----------------------
        energy_quantized = torch.bucketize(energy_pred, self.energy_bins)
        energy_emb = self.energy_emb(energy_quantized)
        output = output + energy_emb

        decoder_output = self.decoder(output, ~feature_mask)
        output = self.mel_linear(decoder_output)
        postnet_output = self.postnet(output) + output

        return AcousticModelOutput(
            pred_features={
                "mel_spectrogram": postnet_output.permute(0,2,1),
                "pitch": pitch_pred.unsqueeze(1),
                "mel_energy": energy_pred.unsqueeze(1),
            },
            outputs={
                "decoder_output": decoder_output,
            }
        )
