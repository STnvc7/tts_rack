from typing import Dict
import torch

from interface.model import E2EModelOutput, DiscriminatorOutput
from interface.loss import LossOutput, E2EModelLoss
from interface.data import DataLoaderOutput

from engine._common.loss.acoustic import mel_spectrogram_l1_loss
from engine._common.loss.adversarial import least_square_generator_loss, feature_matching_loss

def kl_divergence(z_p, logs_q, m_p, logs_p, mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * mask)
    l = kl / torch.sum(mask)
    return l

class VITSLoss(E2EModelLoss):
    def __init__(
        self, 
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        lambda_mel: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_fm: float = 1.0,
        lambda_dur: float = 1.0,
        lambda_kl: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_dur = lambda_dur
        self.lambda_kl = lambda_kl

    def forward(
        self,
        batch: DataLoaderOutput,
        e2e_output: E2EModelOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        assert e2e_output.outputs is not None
        kl_loss = kl_divergence(
            e2e_output.outputs["z_p"], 
            e2e_output.outputs["logs_q"], 
            e2e_output.outputs["m_p"], 
            e2e_output.outputs["logs_p"], 
            batch.feature_mask.unsqueeze(1),
        ) * self.lambda_kl
        
        duration_loss = e2e_output.outputs["duration_loss"] * self.lambda_dur
        
        target = batch.wav.squeeze(1)
        pred = e2e_output.pred.squeeze(1)
        
        reconstruct_loss = mel_spectrogram_l1_loss(
            target, pred, 
            self.sample_rate, self.fft_size, self.hop_size
        ) * self.lambda_mel

        adv_loss = torch.tensor(0.0, device=target.device)
        fm_loss = torch.tensor(0.0, device=target.device)
        for out in discriminator_outputs.values():
            adv_loss += least_square_generator_loss(out.pred) * self.lambda_adv
            fm_loss += feature_matching_loss(out.fmap_target, out.fmap_pred) * self.lambda_fm
        
        total_loss = reconstruct_loss + adv_loss + fm_loss + kl_loss + duration_loss
        
        output = LossOutput(
            total_loss=total_loss, 
            loss_components={
                "generator_adv_loss": adv_loss,
                "feature_matching_loss": fm_loss,
                "reconstruct_loss": reconstruct_loss,
                "kl_loss": kl_loss,
                "duration_loss": duration_loss
            }
        )
        
        return output