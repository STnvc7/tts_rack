from typing import cast

import torch
import torch.nn as nn

from interface.loss import AcousticModelLoss, LossOutput
from interface.data import DataLoaderOutput
from interface.model import AcousticModelOutput

class FastSpeech2Loss(AcousticModelLoss):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, batch: DataLoaderOutput, acoustic_output: AcousticModelOutput) -> LossOutput:
        src_masks = batch.phoneme_id_mask
        mel_masks = batch.feature_mask
        mel_target = batch.features["mel_spectrogram"].permute(0,2,1)
        
        assert acoustic_output.outputs is not None
        
        mel_pred = acoustic_output.outputs["output"]
        mel_pred = mel_pred.masked_select(mel_masks[...,None])
        mel_postnet_pred = acoustic_output.outputs["postnet_output"]
        mel_postnet_pred = mel_postnet_pred.masked_select(mel_masks[...,None])
        mel_target = mel_target.masked_select(mel_masks[...,None])
        
        pitch_pred = acoustic_output.outputs["pitch_pred_norm"]
        pitch_pred = pitch_pred.masked_select(mel_masks)
        pitch_target = acoustic_output.outputs["pitch_raw_norm"]
        pitch_target = pitch_target.masked_select(mel_masks)
        
        energy_pred = acoustic_output.outputs["energy_pred_norm"]
        energy_pred = energy_pred.masked_select(mel_masks)
        energy_target = acoustic_output.outputs["energy_raw_norm"]
        energy_target = energy_target.masked_select(mel_masks)

        assert batch.duration is not None
        log_duration_pred = acoustic_output.outputs["log_duration_pred"]
        log_duration_pred = log_duration_pred.masked_select(src_masks)
        log_duration_target = torch.log(batch.duration.float() + 1)
        log_duration_target = log_duration_target.masked_select(src_masks)

        mel_loss = self.mae_loss(mel_pred, mel_target)
        postnet_mel_loss = self.mae_loss(mel_postnet_pred, mel_target)
        pitch_loss = self.mse_loss(pitch_pred, pitch_target)
        energy_loss = self.mse_loss(energy_pred, energy_target)
        duration_loss = self.mse_loss(log_duration_pred, log_duration_target)

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss

        return LossOutput(
            total_loss=total_loss,
            loss_components={
                "mel_loss": mel_loss,
                "postnet_mel_loss": postnet_mel_loss,
                "pitch_loss": pitch_loss,
                "energy_loss": energy_loss,
                "duration_loss": duration_loss
            }
        )