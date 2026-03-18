from typing import Dict
import torch
import torch.nn.functional as F

from interface.model import E2EModelOutput, DiscriminatorOutput
from interface.loss import LossOutput, E2EModelLoss
from interface.data import DataLoaderOutput

from engine._common.loss.acoustic import mel_spectrogram_l1_loss
from engine._common.loss.adversarial import least_square_generator_loss, feature_matching_loss

class SmallSpeechLoss(E2EModelLoss):
    def __init__(
        self, 
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        lambda_mel: float = 1.0,
        lambda_mu: float = 1.0,
        lambda_prior: float = 1.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.lambda_mel = lambda_mel
        self.lambda_mu = lambda_mu
        self.lambda_prior = lambda_prior
        
    def forward(
        self,
        batch: DataLoaderOutput,
        e2e_output: E2EModelOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        assert e2e_output.outputs is not None
        mu_loss = e2e_output.outputs["mu_loss"] * self.lambda_mu
        prior_loss = e2e_output.outputs["prior_loss"] * self.lambda_prior
        f0_loss = e2e_output.outputs["f0_loss"]
        duration_loss = e2e_output.outputs["duration_loss"]
        total_loss = mu_loss + prior_loss + f0_loss + duration_loss
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
                "mu_loss": mu_loss,
                "prior_loss": prior_loss,
                "f0_loss": f0_loss,
                "duration_loss": duration_loss,
            }
        )
        
        return output