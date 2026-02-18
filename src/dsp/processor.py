import math
import torch
from dsp_board.processor import Processor

class DSPProcessor(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Define your custom methods!
    def continuous_pitch(self, x) -> torch.Tensor:
        f0 = self.pitch(x).squeeze(0)
        cf0 = f0.clone()
        
        # 非ゼロ（有声音）のインデックスを取得
        nz_mask = cf0 != 0
        nz_indices = torch.nonzero(nz_mask).squeeze()
    
        # 全て0、または有声フレームが1つ以下の場合はそのまま返す
        if nz_indices.numel() <= 1:
            return cf0
    
        # 1. 先頭と末尾の埋め合わせ (Edge Padding)
        start_idx = nz_indices[0]
        end_idx = nz_indices[-1]
        
        start_f0 = cf0[start_idx]
        end_f0 = cf0[end_idx]
        
        cf0[:start_idx] = start_f0
        cf0[end_idx+1:] = end_f0
    
        # 2. 内部の0を線形補間 (Linear Interpolation)
        # 元コードの interpolate.interp1d 相当の処理
        
        # 全フレームのインデックス
        grid = torch.arange(len(cf0), device=cf0.device)
        
        # 各時点がどの非ゼロ区間の間にあるかを探索
        # nz_indices は既知の点（x座標）、grid は求めたい点
        idx_bins = torch.searchsorted(nz_indices, grid)
        idx_bins = torch.clamp(idx_bins, min=1, max=len(nz_indices) - 1)
        
        # 左側の既知の点 (x0, y0)
        x0 = nz_indices[idx_bins - 1]
        y0 = cf0[x0]
        
        # 右側の既知の点 (x1, y1)
        x1 = nz_indices[idx_bins]
        y1 = cf0[x1]
        
        # 線形補間の計算: y = y0 + (x - x0) * slope
        # slope = (y1 - y0) / (x1 - x0)
        slope = (y1 - y0) / (x1 - x0).to(cf0.dtype) # 除算のために型合わせ
        interpolated = y0 + slope * (grid - x0).to(cf0.dtype)
        cf0[start_idx:end_idx+1] = interpolated[start_idx:end_idx+1]
    
        return cf0.unsqueeze(0)