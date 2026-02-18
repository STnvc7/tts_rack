import torch
import torch.nn as nn

class Shuffle1d(nn.Module):
    def __init__(self, n_groups):
        super().__init__()
        self.n_groups = n_groups
    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B, self.n_groups, C//self.n_groups, L)
        x = x.permute(0,2,1,3)
        x = x.reshape(B,C,L)
        return x