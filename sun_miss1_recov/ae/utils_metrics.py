"""utils_metrics.py — PSNR (MISS0와 동일 정의)"""
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    val = 10 * torch.log10((max_val ** 2) / mse)
    return val.mean()


def masked_psnr_torch(pred, target, mask, max_val=1.0, eps=1e-10):
    diff2 = (pred - target) ** 2
    region_sum = (diff2 * mask).sum(dim=[1, 2, 3])
    region_cnt = mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
    mse = (region_sum / region_cnt).clamp(min=eps)
    val = 10 * torch.log10((max_val ** 2) / mse)
    return val.mean()