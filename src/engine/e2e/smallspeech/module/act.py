import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, scale, shift):
        return F.gelu(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, scale, shift):
        return x * torch.sigmoid(x)

class PSwish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, scale, shift):
        return x * torch.sigmoid(scale * x)
        
class Tangma(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, scale, shift):
        return x * torch.tanh(x + shift) + scale * x

class Activation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        if act == "gelu":
            self.act_fn = GELU()
        elif act == "swish":
            self.act_fn = Swish()
        elif act == "pswish":
            self.act_fn = PSwish()
        elif act == "tangma":
            self.act_fn = Tangma()
        else:
            raise ValueError(f"Unknown activation: {act}")
    def forward(self, x, scale, shift):
        return self.act_fn(x, scale, shift)