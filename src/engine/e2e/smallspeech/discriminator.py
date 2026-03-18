from typing import Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from interface.model import Discriminator, GeneratorOutput, DiscriminatorOutput, E2EModelOutput

class Discriminator1d(nn.Module):
    def __init__(self, c_channels, kernel_size, dilation, conditional=True):
        super().__init__()
        self.pre = nn.Conv1d(1, 128, 1)
        self.conditional = conditional
        if conditional:
            self.c_pre = nn.Conv1d(c_channels, 128, 1)
        
        padding = (dilation * (kernel_size - 1)) // 2
        self.convs = nn.ModuleList([
            nn.Conv1d(128, 128, kernel_size, padding=padding, dilation=dilation),
            nn.Conv1d(128, 256, kernel_size, padding=padding, dilation=dilation),
            nn.Conv1d(256, 512, kernel_size, padding=padding, dilation=dilation),
            nn.Conv1d(512, 1024, kernel_size, padding=padding, dilation=dilation),
            nn.Conv1d(1024, 1024, kernel_size, padding=padding, dilation=dilation),
        ])
        self.post = nn.Conv1d(1024, 1, 1)
        
    def forward(self, x, c, mask):
        z = self.pre(x)
        if self.conditional:
            z = z + self.c_pre(c)
        fmap = [z]
        for conv in self.convs:
            z = conv(z) * mask
            z = F.leaky_relu(z, 0.1)
            fmap.append(z)
        z = self.post(z)
        return z, fmap
        
class F0Discriminator(Discriminator):
    def __init__(self, c_channels, kernel_sizes, dilations, conditional=True):
        super().__init__()
        self.discs = nn.ModuleList([
            Discriminator1d(c_channels, kernel_size=k, dilation=d, conditional=conditional)
            for k, d in zip(kernel_sizes, dilations)
        ])
        
    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput],
        mode: Literal["generator", "discriminator"]
    ) -> DiscriminatorOutput:
        assert generator_output.outputs is not None
        
        target_f0 = generator_output.outputs["log_f0_target"]
        pred_f0 = generator_output.outputs["log_f0_pred"]
        if mode == "discriminator":
            pred_f0 = pred_f0.detach()
            
        c = generator_output.outputs["z_feature"].detach()
        mask = generator_output.outputs["y_mask"]        
        
        outs_target = []
        outs_pred = []
        fmaps_target = []
        fmaps_pred = []
        for disc in self.discs:
            out_target, fmap_target = disc(target_f0, c, mask)
            out_pred, fmap_pred = disc(pred_f0, c, mask)
            outs_target.append(out_target)
            outs_pred.append(out_pred)
            fmaps_target.append(fmap_target)
            fmaps_pred.append(fmap_pred)

        return DiscriminatorOutput(
            target=outs_target, 
            pred=outs_pred, 
            fmap_target=fmaps_target, 
            fmap_pred=fmaps_pred
        )
        
      
class DurationDiscriminator(Discriminator):
    def __init__(self, c_channels, kernel_sizes, dilations, conditional=True):
        super().__init__()
        self.discs = nn.ModuleList([
            Discriminator1d(c_channels, kernel_size=k, dilation=d, conditional=conditional)
            for k, d in zip(kernel_sizes, dilations)
        ])
        
    def forward(
        self, 
        target: torch.Tensor, 
        generator_output: Union[GeneratorOutput, E2EModelOutput],
        mode: Literal["generator", "discriminator"]
    ) -> DiscriminatorOutput:
        assert generator_output.outputs is not None
        
        target_d = generator_output.outputs["duration_target"]
        pred_d = generator_output.outputs["duration_pred"]
        if mode == "discriminator":
            pred_d = pred_d.detach()
        
        c = generator_output.outputs["z_text"].detach()
        mask = generator_output.outputs["x_mask"]
        
        outs_target = []
        outs_pred = []
        fmaps_target = []
        fmaps_pred = []
        for disc in self.discs:
            out_target, fmap_target = disc(target_d, c, mask)
            out_pred, fmap_pred = disc(pred_d, c, mask)
            outs_target.append(out_target)
            outs_pred.append(out_pred)
            fmaps_target.append(fmap_target)
            fmaps_pred.append(fmap_pred)
            
        return DiscriminatorOutput(
            target=outs_target, 
            pred=outs_pred, 
            fmap_target=fmaps_target, 
            fmap_pred=fmaps_pred
        )
      
