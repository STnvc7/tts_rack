import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, channels, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        else:
            self.norm = nn.LayerNorm(channels)
        
    def forward(self, x, scale=None, shift=None):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        if self.adaptive:
            x = x * scale + shift
        return x
      