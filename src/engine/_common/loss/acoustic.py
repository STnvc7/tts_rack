from typing import List, Tuple
import torch
import torch.nn.functional as F
import math
from dsp_board.features import spectrogram, mel_spectrogram

def spectrogram_l1_loss(
    target: torch.Tensor, 
    pred: torch.Tensor, 
    fft_size: int=1024, 
    hop_size: int=256
) -> torch.Tensor:
    
    fn = lambda x: spectrogram(x, fft_size, hop_size, log=True)
    target = fn(target)
    pred = fn(pred)
    loss = F.l1_loss(target, pred)
    return loss

def multi_resolution_stft_loss(
    target: torch.Tensor, 
    pred: torch.Tensor, 
    fft_sizes: List[int]=[1024, 2048], 
    hop_sizes: List[int]=[256, 512]
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=target.device)
    for fft_size, hop_size in zip(fft_sizes, hop_sizes):
        loss = loss + spectrogram_l1_loss(target, pred, fft_size, hop_size)
    return loss


def mel_spectrogram_l1_loss(
    target: torch.Tensor, 
    pred: torch.Tensor, 
    sample_rate: int,
    fft_size=1024, 
    hop_size=256, 
    n_mels=80
) -> torch.Tensor:
    fn = lambda x: mel_spectrogram(x, sample_rate, fft_size, hop_size, n_mels, log=True)
    target = fn(target)
    pred = fn(pred)
    loss = F.l1_loss(target, pred)
    return loss
    

def phase_loss(
    phase_target: torch.Tensor, 
    phase_pred: torch.Tensor, 
    fft_size: int=1024
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # calculate group delay --------------------- 
    group_delay_matrix = (
        torch.triu(torch.ones(fft_size // 2 + 1, fft_size // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(fft_size // 2 + 1, fft_size // 2 + 1), diagonal=2)
        - torch.eye(fft_size // 2 + 1)
    )
    group_delay_matrix = group_delay_matrix.to(phase_target.device)

    group_delay_target = torch.matmul(phase_target.permute(0, 2, 1), group_delay_matrix)
    group_delay_pred = torch.matmul(phase_pred.permute(0, 2, 1), group_delay_matrix)

    # caluculate phase time difference ----------
    phase_time_diff_matrix = (
        torch.triu(torch.ones(phase_target.shape[-1], phase_target.shape[-1]), diagonal=1)
        - torch.triu(torch.ones(phase_target.shape[-1], phase_target.shape[-1]), diagonal=2)
        - torch.eye(phase_target.shape[-1])
    )
    phase_time_diff_matrix = phase_time_diff_matrix.to(phase_target.device)

    phase_time_diff_target = torch.matmul(phase_target, phase_time_diff_matrix)
    phase_time_diff_pred = torch.matmul(phase_pred, phase_time_diff_matrix)
    
    # calculate phase losses ----------------------
    anti_wrapping_function = lambda p: torch.abs(p - torch.round(p / (2 * math.pi)) * 2 * math.pi)
    instantaneous_phase_loss = torch.mean(anti_wrapping_function(phase_target - phase_pred))
    group_delay_loss = torch.mean(anti_wrapping_function(group_delay_target - group_delay_pred))
    phase_time_difference_loss = torch.mean(anti_wrapping_function(phase_time_diff_target - phase_time_diff_pred))

    return instantaneous_phase_loss, group_delay_loss, phase_time_difference_loss