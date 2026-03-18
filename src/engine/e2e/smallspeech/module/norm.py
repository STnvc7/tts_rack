import torch.nn as nn

class LayerNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x
      
class AdaptiveNorm1d(nn.Module):
  def __init__(self, channels):
      super().__init__()
      self.norm = nn.LayerNorm(channels, elementwise_affine=False)
      
  def forward(self, x, scale, shift):
      x = x.transpose(1, 2)
      x = self.norm(x)
      x = x.transpose(1, 2)
      x = x * scale + shift
      return x