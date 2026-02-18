from typing import List, Dict
import torch
import torch.nn.functional as F
from dsp_board.features import mel_spectrogram

from interface.model import GeneratorOutput, DiscriminatorOutput
from interface.loss import LossOutput, GeneratorLoss
from interface.data import DataLoaderOutput
from engine._common.loss.acoustic import mel_spectrogram_l1_loss
from engine._common.loss.adversarial import feature_matching_loss, least_square_generator_loss

class HiFiGANGeneratorLoss(GeneratorLoss):
    def __init__(
        self,
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        lambda_mel=45, 
        lambda_adv=1, 
        lambda_fm=2
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels

        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
    
    def forward(
        self,
        batch: DataLoaderOutput,
        generator_output: GeneratorOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        target = batch.wav.squeeze(1)
        pred = generator_output.pred.squeeze(1)
        
        reconstruct_loss = mel_spectrogram_l1_loss(
            target, pred, 
            self.sample_rate, self.fft_size, self.hop_size
        ) * self.lambda_mel

        adv_loss = torch.tensor(0.0, device=target.device)
        fm_loss = torch.tensor(0.0, device=target.device)
        for out in discriminator_outputs.values():
            adv_loss += least_square_generator_loss(out.pred) * self.lambda_adv
            fm_loss += feature_matching_loss(out.fmap_target, out.fmap_pred) * self.lambda_fm
        
        total_loss = reconstruct_loss + adv_loss + fm_loss
        
        output = LossOutput(
            total_loss=total_loss, 
            loss_components={
                "generator_adv_loss": adv_loss,
                "feature_matching_loss": fm_loss,
                "reconstruct_loss": reconstruct_loss
            }
        )
        
        return output