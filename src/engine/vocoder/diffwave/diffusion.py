import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
    beta: torch.Tensor
    noise_level: torch.Tensor
    def __init__(
        self,
        model: nn.Module,
        upsample_rate,
        beta_min=1e-6,
        beta_max=0.01,
        beta_step=1000,
    ):
        super().__init__()
        self.model = model
        self.upsample_rate = upsample_rate
        self.beta_step = beta_step
        beta = torch.linspace(beta_min, beta_max, beta_step)
        self.register_buffer("beta", beta)
        self.register_buffer("noise_level", torch.cumprod(1 - beta, dim=0))
        self.loss_fn = nn.L1Loss()
        
    @torch.inference_mode()
    def forward(self, mel, noise_schedule=None):
        if noise_schedule is None:
            infer_beta = self.beta
        else:
            infer_beta = noise_schedule
    
        alpha = 1 - self.beta
        alpha_cum = torch.cumprod(alpha, dim=0)
    
        infer_alpha = 1 - infer_beta
        infer_alpha_cum = torch.cumprod(infer_alpha, dim=0)
    
        T = []
        for s in range(len(infer_beta)):
            for t in range(len(self.beta) - 1):
                if alpha_cum[t+1] <= infer_alpha_cum[s] <= alpha_cum[t]:
                    twiddle = (alpha_cum[t]**0.5 - infer_alpha_cum[s]**0.5) / (alpha_cum[t]**0.5 - alpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = torch.tensor(T, dtype=torch.float32, device=mel.device)
    
        B, _, L = mel.shape
        audio = torch.randn(B, 1, L*self.upsample_rate, device=mel.device)
        for n in range(len(infer_alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = infer_beta[n] / (1 - alpha_cum[n])**0.5
            t = T[n].unsqueeze(0).expand(B)
            audio = c1 * (audio - c2 * self.model(audio, t, mel).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - infer_alpha_cum[n-1]) / (1.0 - infer_alpha_cum[n]) * infer_beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
    
    def compute_loss(self, wav, mel):
        B, _, _ = wav.shape
        N = len(self.beta)
        t = torch.randint(0, N, [B], device=wav.device)
        noise_scale = self.noise_level[t][:, None, None]
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(wav)
        noisy_audio = noise_scale_sqrt * wav + (1.0 - noise_scale)**0.5 * noise
        predicted = self.model(noisy_audio, t, mel)
        loss = self.loss_fn(noise, predicted)
        return loss
        