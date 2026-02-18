from typing import Dict
import torch

from interface.model import E2EModelOutput, DiscriminatorOutput
from interface.loss import LossOutput, E2EModelLoss
from interface.data import DataLoaderOutput

from engine._common.loss.acoustic import mel_spectrogram_l1_loss
from engine._common.loss.adversarial import least_square_generator_loss, feature_matching_loss

class DDSpeechLoss(E2EModelLoss):
    def __init__(
        self, 
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        lambda_mel: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.lambda_mel = lambda_mel

    def forward(
        self,
        batch: DataLoaderOutput,
        e2e_output: E2EModelOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        assert e2e_output.outputs is not None

        duration_loss = e2e_output.outputs["duration_loss"]
        f0_cfm_loss = e2e_output.outputs["f0_cfm_loss"]
        mel_mu_loss = e2e_output.outputs["mel_mu_loss"]
        total_loss = duration_loss + f0_cfm_loss + mel_mu_loss
        
        target = batch.wav.squeeze(1)
        pred = e2e_output.pred.squeeze(1)
        
        reconstruct_loss = mel_spectrogram_l1_loss(
            target, pred, 
            self.sample_rate, self.fft_size, self.hop_size
        ) * self.lambda_mel

        adv_loss = torch.tensor(0.0, device=target.device)
        fm_loss = torch.tensor(0.0, device=target.device)
        for out in discriminator_outputs.values():
            adv_loss += least_square_generator_loss(out.pred)
            fm_loss += feature_matching_loss(out.fmap_target, out.fmap_pred)
        
        total_loss += reconstruct_loss + adv_loss + fm_loss
        
        output = LossOutput(
            total_loss=total_loss, 
            loss_components={
                "generator_adv_loss": adv_loss,
                "feature_matching_loss": fm_loss,
                "reconstruct_loss": reconstruct_loss,
                "duration_loss": duration_loss,
                "f0_cfm_loss": f0_cfm_loss,
                "mel_mu_loss": mel_mu_loss,
            }
        )
        
        return output