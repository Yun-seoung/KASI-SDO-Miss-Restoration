"""utils_metrics.py — PSNR (Tiny ImageNet 실습 때 utils.py와 동일 정의, [0,1] 기준)"""
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """pred, target: (B,1,H,W), 0~1 범위. 배치 평균 PSNR(dB)."""
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    val = 10 * torch.log10((max_val ** 2) / mse)
    return val.mean()


def masked_psnr_torch(pred, target, mask, max_val=1.0, eps=1e-10):
    """마스킹 영역만의 PSNR (배치 단위, baseline의 masked_psnr과 동일 철학)."""
    diff2 = (pred - target) ** 2
    region_sum = (diff2 * mask).sum(dim=[1, 2, 3])
    region_cnt = mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
    mse = (region_sum / region_cnt).clamp(min=eps)
    val = 10 * torch.log10((max_val ** 2) / mse)
    return val.mean()