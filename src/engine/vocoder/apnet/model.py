from typing import Dict, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from dsp_board.transforms import istft

from interface.model import Generator, GeneratorOutput
from interface.feature import AcousticFeature
from interface.loggable import Heatmap

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
    
class ASPResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ASPResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class PSPResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(PSPResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class APNetGenerator(Generator):
    def __init__(
        self,
        n_mels,
        fft_size,
        hop_size,
        asp_channel,
        asp_input_conv_kernel_size,
        asp_resblock_kernel_sizes,
        asp_resblock_dilation_sizes,
        asp_output_conv_kernel_size,
        psp_channel,
        psp_input_conv_kernel_size,
        psp_resblock_kernel_sizes,
        psp_resblock_dilation_sizes,
        psp_output_real_conv_kernel_size,
        psp_output_imag_conv_kernel_size,
    ):
        super(APNetGenerator, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        
        self.asp_num_kernels = len(asp_resblock_kernel_sizes)
        self.psp_num_kernels = len(psp_resblock_kernel_sizes)

        self.asp_input_conv = weight_norm(Conv1d(
            n_mels, asp_channel, asp_input_conv_kernel_size, padding=get_padding(asp_input_conv_kernel_size)
        ))
        self.psp_input_conv = weight_norm(Conv1d(
            n_mels, psp_channel, psp_input_conv_kernel_size, padding=get_padding(psp_input_conv_kernel_size)
        ))

        self.asp_resnet = nn.ModuleList()
        for k, d in zip(asp_resblock_kernel_sizes, asp_resblock_dilation_sizes):
            self.asp_resnet.append(ASPResBlock(asp_channel, k, d))

        self.psp_resnet = nn.ModuleList()
        for k, d in zip(psp_resblock_kernel_sizes, psp_resblock_dilation_sizes):
            self.psp_resnet.append(PSPResBlock(psp_channel, k, d))

        self.asp_output_conv = weight_norm(Conv1d(
            asp_channel, fft_size//2+1, asp_output_conv_kernel_size, padding=get_padding(asp_output_conv_kernel_size)
        ))
        self.psp_output_real_conv = weight_norm(Conv1d(
            psp_channel, fft_size//2+1, psp_output_real_conv_kernel_size, padding=get_padding(psp_output_real_conv_kernel_size)
        ))
        self.psp_output_imag_conv = weight_norm(Conv1d(
            psp_channel, fft_size//2+1, psp_output_imag_conv_kernel_size, padding=get_padding(psp_output_imag_conv_kernel_size)))

        self.asp_output_conv.apply(init_weights)
        self.psp_output_real_conv.apply(init_weights)
        self.psp_output_imag_conv.apply(init_weights)

    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        
        mel = input_feature["mel_spectrogram"]
        logamp = self.asp_input_conv(mel)
        logamps = 0
        for m in self.asp_resnet:
            logamps += m(logamp)
        logamp = logamps / self.asp_num_kernels
        logamp = F.leaky_relu(logamp)
        logamp = self.asp_output_conv(logamp)

        pha = self.psp_input_conv(mel)
        phas = 0
        for m in self.psp_resnet:
            phas += m(pha)
        pha = phas / self.psp_num_kernels
        pha = F.leaky_relu(pha)   
        real = self.psp_output_real_conv(pha)
        imag = self.psp_output_imag_conv(pha)

        pha = torch.atan2(imag, real)

        real = torch.exp(logamp)*torch.cos(pha)
        imag = torch.exp(logamp)*torch.sin(pha)
        spc = torch.complex(real, imag)
        audio = istft(spc, self.fft_size, self.hop_size)
        
        output = GeneratorOutput(
            pred=audio.unsqueeze(1),
            outputs={
                "real": real,
                "imag": imag,
                "logamp": logamp,
                "phase": pha,
            },
            loggable_outputs={
                "mag": Heatmap(logamp.squeeze()),
                "phase": Heatmap(pha.squeeze()),
            }
        )
        return output
