import torch
import torch.nn as nn

class ShuffleConvNeXt(nn.Module):
    def __init__(
        self,
        channel,
        h_channel,
        n_groups,
        kernel_sizes,
        activation=nn.GELU(),
    ):
        super().__init__()
        
        n_layer = len(kernel_sizes)
        scale = 1 / n_layer
        layer = [
            ShuffleConvNeXtBlock(
                channel=channel, 
                h_channel=h_channel,
                n_groups=n_groups,
                kernel_size=kernel_sizes[i], 
                scale=scale,
                activation=activation,
            )
            for i in range(n_layer)
        ]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Shuffle(nn.Module):
    def __init__(self, n_groups):
        super().__init__()
        self.n_groups = n_groups
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, self.n_groups, C//self.n_groups, H, W)
        x = x.permute(0,2,1,3,4)
        x = x.reshape(B, C, H, W)
        return x

class LayerNorm(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.norm = nn.LayerNorm(channel)
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x

class ShuffleConvNeXtBlock(nn.Module):
    def __init__(self, channel, h_channel, n_groups, kernel_size, scale, activation=nn.GELU()):
        super().__init__()
        self.dw_conv = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channel)
        self.norm = LayerNorm(channel)
        self.pw_conv1 = nn.Conv2d(channel, h_channel, 1, groups=n_groups)
        self.shuffle = Shuffle(n_groups)
        self.act = activation
        self.pw_conv2 = nn.Conv2d(h_channel, channel, 1, groups=n_groups)
        self.scale = nn.Parameter(torch.full(size=(1, channel, 1, 1), fill_value=scale), requires_grad=True)
        
    def forward(self, x):
        
        res = x
        
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.shuffle(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x
