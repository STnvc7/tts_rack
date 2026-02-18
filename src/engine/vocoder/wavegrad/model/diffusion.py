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
        
        noise_level = torch.cumprod(1 - beta, dim=0)**0.5
        noise_level = torch.cat([torch.tensor([1.0]), noise_level], dim=0)
        self.register_buffer("noise_level", noise_level)
        self.loss_fn = nn.L1Loss()
        
    @torch.inference_mode()
    def forward(self, mel, noise_schedule=None):
        if noise_schedule is None:
            beta = self.beta
        else:
            beta = noise_schedule
        N = beta.shape[0]
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)
        noise_scale = alpha_cum ** 0.5
        
        B, _, T = mel.shape
        y_t_1 = torch.randn(B, 1, T*self.upsample_rate, device=mel.device)
        for n in range(N-1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
            y_t_1 = c1 * (y_t_1 - c2 * self.model(y_t_1, mel, noise_scale[n].unsqueeze(0)))
            if n > 0:
                noise = torch.randn_like(y_t_1)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                y_t_1 = y_t_1 + sigma * noise
            y_t_1 = torch.clamp(y_t_1, -1.0, 1.0)
        return y_t_1
    
    def compute_loss(self, wav, mel):
        B, _, T = wav.shape
        s = torch.randint(1, self.beta_step + 1, [B], device=wav.device)
        l_a, l_b = self.noise_level[s-1], self.noise_level[s]
        noise_scale = l_a + torch.rand(B, device=wav.device) * (l_b - l_a)
        noise = torch.randn_like(wav)
        noisy_audio = noise_scale[:,None,None] * wav + (1.0 - noise_scale[:,None,None]**2)**0.5 * noise
        predicted = self.model(noisy_audio, mel, noise_scale)
        loss = self.loss_fn(noise, predicted)
        return loss