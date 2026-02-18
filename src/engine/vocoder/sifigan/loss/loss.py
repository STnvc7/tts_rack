from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dsp_board.features import pitch

from interface.model import GeneratorOutput, DiscriminatorOutput
from interface.loss import LossOutput, GeneratorLoss
from interface.data import DataLoaderOutput
from engine._common.loss.adversarial import least_square_generator_loss, feature_matching_loss
from engine._common.loss.acoustic import mel_spectrogram_l1_loss

from .reg import ResidualLoss

class SiFiGANGeneratorLoss(GeneratorLoss):
    def __init__(
        self,
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        lambda_mel=45, 
        lambda_adv=1, 
        lambda_fm=2, 
        lambda_reg=1
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_reg = lambda_reg

        self.reg_loss = ResidualLoss(
            sample_rate=sample_rate,
            fft_size=2048,
            hop_size=hop_size,
            f0_floor=100,
            f0_ceil=840,
            n_mels=n_mels,
            fmin=0,
            fmax=None,
            power=False,
            elim_0th=True,
        )

    def forward(
        self,
        batch: DataLoaderOutput,
        generator_output: GeneratorOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        target = batch.wav.squeeze(1)
        generated = generator_output.pred.squeeze(1)
        source = generator_output.outputs["source"].data
        
        # ↓ too slow....
        if self.lambda_reg != 0:
            f0 = torch.stack([pitch(t, self.sample_rate, self.hop_size, "harvest").squeeze(0) for t in target]).unsqueeze(1)
            reg_loss = self.reg_loss(source, target.unsqueeze(1), f0) * self.lambda_reg
        else:
            reg_loss = torch.tensor(0.0, device=target.device)
            
        reconstruct_loss = mel_spectrogram_l1_loss(
            target, generated, 
            self.sample_rate, self.fft_size, self.hop_size
        ) * self.lambda_mel

        adv_loss = torch.tensor(0.0, device=target.device)
        fm_loss = torch.tensor(0.0, device=target.device)
        for out in discriminator_outputs.values():
            adv_loss += least_square_generator_loss(out.pred) * self.lambda_adv
            fm_loss += feature_matching_loss(out.fmap_target, out.fmap_pred) * self.lambda_fm
        
        total_loss = reconstruct_loss + adv_loss + fm_loss + reg_loss
        
        output = LossOutput(
            total_loss=total_loss, 
            loss_components={
                "generator_adv_loss": adv_loss,
                "feature_matching_loss": fm_loss,
                "reconstruct_loss": reconstruct_loss,
                "regularization_loss": reg_loss
            }
        )
        
        return output