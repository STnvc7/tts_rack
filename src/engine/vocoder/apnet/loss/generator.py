from typing import Dict
import torch
import torch.nn.functional as F

from interface.model import GeneratorOutput, DiscriminatorOutput
from interface.loss import LossOutput, GeneratorLoss
from interface.data import DataLoaderOutput

from engine._common.loss.adversarial import least_square_generator_loss, hinge_generator_loss, feature_matching_loss
from engine._common.loss.acoustic import phase_loss, mel_spectrogram_l1_loss

def stft(x, fft_size, hop_size):
    pad_size = (fft_size - hop_size)//2
    _x = F.pad(x, (pad_size, pad_size))
    window = torch.hann_window(fft_size)
        
    spc = torch.stft(
        _x, 
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window.to(x.device),
        center=False,
        onesided=True,
        normalized=True,
        return_complex=True
    )
    return spc

class APNetGeneratorLoss(GeneratorLoss):
    def __init__(
        self,
        sample_rate,
        fft_size,
        hop_size,
        n_mels,
        eps=1e-8,
        adv_loss="least_square",
        lambda_amp=45,
        lambda_phase=100,
        lambda_stft=20,
        lambda_stft_complex=2.25,
        lambda_mel=45, 
        lambda_disc: Dict[str, float]= {"mpd": 1, "mrd": 1}
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.eps = eps
        
        if adv_loss == "least_square":
            self.adv_loss = least_square_generator_loss
        elif adv_loss == "hinge":
            self.adv_loss = hinge_generator_loss
        else:
            raise ValueError(f"Invalid adversarial loss: {adv_loss}")

        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.lambda_stft = lambda_stft
        self.lambda_stft_complex = lambda_stft_complex
        self.lambda_mel = lambda_mel
        self.lambda_disc = lambda_disc
    
    def forward(
        self,
        batch: DataLoaderOutput,
        generator_output: GeneratorOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        target = batch.wav.squeeze(1)
        spc_target = stft(target, self.fft_size, self.hop_size)
        amp_target = spc_target.abs().clip(min=self.eps).log()
        phase_target = spc_target.angle()

        pred = generator_output.pred.squeeze(1)
        spc_pred_final = stft(pred, self.fft_size, self.hop_size)
        
        spc_generated_real = generator_output.outputs["real"]
        spc_generated_imag = generator_output.outputs["imag"]
        amp_generated = generator_output.outputs["logamp"]
        phase_generated = generator_output.outputs["phase"]
        
        # amplitude loss ------------
        amp_loss = F.mse_loss(amp_target, amp_generated) * self.lambda_amp
        
        # phase loss ----------------
        phase_losses = phase_loss(phase_target, phase_generated, self.fft_size)
        instataneous_phase_loss = phase_losses[0]
        group_delay_loss = phase_losses[1]
        phase_time_difference_loss = phase_losses[2]
        phase_loss_total = instataneous_phase_loss + group_delay_loss + phase_time_difference_loss
        phase_loss_total = phase_loss_total * self.lambda_phase
        
        # stft loss -----------------
        consistency_loss = torch.mean(torch.mean(
            (spc_generated_real - spc_pred_final.real)**2 + (spc_generated_imag - spc_pred_final.imag)**2,
            (1, 2)
        ))
        stft_real_loss = F.l1_loss(spc_target.real, spc_generated_real)
        stft_imag_loss = F.l1_loss(spc_target.imag, spc_generated_imag)
        stft_complex_loss = (stft_real_loss + stft_imag_loss) * self.lambda_stft_complex
        stft_loss_total = (stft_complex_loss + consistency_loss) * self.lambda_stft
        
        
        # mel spectrogram loss ------
        mel_loss = mel_spectrogram_l1_loss(
            target, pred, 
            self.sample_rate, self.fft_size, self.hop_size
        ) * self.lambda_mel

        adv_loss = 0
        fm_loss = 0
        for key, out in discriminator_outputs.items():
            _adv_loss = self.adv_loss(out.pred)
            adv_loss += _adv_loss * self.lambda_disc[key]
            
            _fm_loss = feature_matching_loss(out.fmap_target, out.fmap_pred)
            fm_loss += _fm_loss * self.lambda_disc[key]
            
        total_loss = amp_loss + phase_loss_total + stft_loss_total + mel_loss + adv_loss + fm_loss
        
        
        output = LossOutput(
            total_loss=total_loss, 
            loss_components={
                "generator_adv_loss": adv_loss,
                "feature_matching_loss": fm_loss,
                "amplitude_loss": amp_loss,
                "instantaneous_phase_loss": instataneous_phase_loss,
                "group_delay_loss": group_delay_loss,
                "phase_time_difference_loss": phase_time_difference_loss,
                "consistency_loss": consistency_loss,
                "stft_complex_loss": stft_complex_loss,
                "mel_loss": mel_loss
            }
        )
        
        return output