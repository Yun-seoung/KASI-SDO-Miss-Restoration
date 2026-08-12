"""
restore_cubic.py — baseline/AE/VAE/LaMa/Palette 공용 복원·오차 함수 모음
(MISS0/MISS1 동일 파일)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve


def restore_vertical_cubic(img2d, mask2d, clip_to_input_range=True, local_margin=150):
    """
    [버그 수정] 기존 코드는 한 열(column) 안의 마스킹 안 된 모든 행(최대 4000여 개)을
    전부 지나가도록 전역(global) cubic spline을 맞춘 뒤 손상 구간에서 평가했음.
    태양 이미지는 흑점·플레어 등 국소적 밝기 변화가 열 전체에 걸쳐 많아서,
    이 모든 지점을 정확히 통과하는 전역 곡선은 극심하게 요동치며(overshoot),
    그 요동이 하필 손상 구간에서 나타나 복원 결과가 노이즈처럼 튀는 원인이 됨
    (실제로 이전 로그들에서 반복적으로 보였던 "overshoot 5만~12만 픽셀"이 그 증거).

    수정: 손상 구간 바로 위아래 local_margin(기본 150행) 범위의 국소 점들로만
    cubic spline을 맞춤 -> 먼 거리의 무관한 특징(다른 흑점 등)이 곡선에 영향을 못 주게 함.

    img2d : (H, W) 손상된 이미지 배열
    mask2d: (H, W) {0,1}, 1=손상(복원 대상)
    local_margin: 손상 구간 위아래로 몇 행까지를 "국소 이웃"으로 볼지
    clip_to_input_range: True면 복원값을 원본(손상 전 알려진 픽셀) 범위 안으로 clip.
    반환  : (H, W) 복원된 배열 (float32)
    """
    h, w = img2d.shape
    out = img2d.copy().astype(np.float32)
    rows = np.arange(h)

    damaged_cols = np.where(mask2d.any(axis=0))[0]
    known_mask = mask2d == 0

    if known_mask.any():
        clip_min = img2d[known_mask].min()
        clip_max = img2d[known_mask].max()
    else:
        clip_min, clip_max = img2d.min(), img2d.max()

    n_overshoot = 0

    for col in damaged_cols:
        col_mask = mask2d[:, col] > 0
        damaged_row_idx = rows[col_mask]
        if len(damaged_row_idx) == 0:
            continue

        # 손상 구간의 위아래 local_margin 범위만 "국소 이웃"으로 사용
        gap_lo, gap_hi = damaged_row_idx.min(), damaged_row_idx.max()
        window_lo = max(0, gap_lo - local_margin)
        window_hi = min(h, gap_hi + local_margin + 1)

        local_rows = rows[window_lo:window_hi]
        local_mask = col_mask[window_lo:window_hi]
        known_rows = local_rows[~local_mask]
        known_vals = img2d[window_lo:window_hi, col][~local_mask]

        if len(known_rows) < 4:
            continue

        cs = CubicSpline(known_rows, known_vals, extrapolate=True)
        interp_vals = cs(damaged_row_idx)

        if clip_to_input_range:
            overshoot = (interp_vals < clip_min) | (interp_vals > clip_max)
            n_overshoot += overshoot.sum()
            interp_vals = np.clip(interp_vals, clip_min, clip_max)

        out[damaged_row_idx, col] = interp_vals

    if clip_to_input_range and n_overshoot > 0:
        print(f"    [info] overshoot 발생 픽셀 {n_overshoot}개 -> clip 처리됨 "
              f"(범위: [{clip_min:.2f}, {clip_max:.2f}])")

    return out


def apply_damage(img2d, mask2d, fill_value=0.0):
    """정상 이미지에 마스크를 적용해 손상 이미지 생성 (baseline 입력용)."""
    damaged = img2d.copy().astype(np.float32)
    damaged[mask2d > 0] = fill_value
    return damaged


def masked_l1(orig, restored, mask2d):
    """마스킹 영역만의 L1 오차 평균 (raw DN 단위)."""
    diff = np.abs(orig.astype(np.float32) - restored.astype(np.float32))
    region = mask2d > 0
    if region.sum() == 0:
        return np.nan
    return float(diff[region].mean())


def masked_rmse(orig, restored, mask2d):
    """마스킹 영역만의 RMSE (raw DN 단위)."""
    diff = (orig.astype(np.float32) - restored.astype(np.float32)) ** 2
    region = mask2d > 0
    if region.sum() == 0:
        return np.nan
    return float(np.sqrt(diff[region].mean()))


def normalize_01(img2d, ref_min=None, ref_max=None, eps=1e-8):
    """이미지를 [0,1]로 min-max 정규화."""
    if ref_min is None:
        ref_min = float(img2d.min())
    if ref_max is None:
        ref_max = float(img2d.max())
    denom = max(ref_max - ref_min, eps)
    return (img2d.astype(np.float32) - ref_min) / denom


def masked_psnr(orig, restored, mask2d, max_val=1.0, eps=1e-10):
    """
    마스킹 영역만의 PSNR (dB).
    AE/VAE 실습 때(utils.py의 psnr())와 동일한 정의:
      - 이미지를 [0,1]로 정규화 후 max_val=1.0 기준으로 계산
    """
    region = mask2d > 0
    if region.sum() == 0:
        return np.nan

    o_min, o_max = float(orig.min()), float(orig.max())
    orig_n = normalize_01(orig, o_min, o_max)
    restored_n = normalize_01(restored, o_min, o_max)

    mse = ((orig_n - restored_n)[region] ** 2).mean()
    mse = max(mse, eps)
    return float(10 * np.log10((max_val ** 2) / mse))


def masked_region_stats(orig, mask2d):
    """마스킹 영역의 원본 픽셀 통계 (오차의 상대적 크기 판단용)."""
    region = mask2d > 0
    if region.sum() == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    vals = orig.astype(np.float32)[region]
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _gaussian_window(size=11, sigma=1.5):
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def _ssim_map(img1, img2, window, data_range=1.0):
    """AE/VAE Tiny ImageNet 실습 때 utils.py의 ssim()과 동일한 정의(11x11 가우시안 윈도우)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = fftconvolve(img1, window, mode="valid")
    mu2 = fftconvolve(img2, window, mode="valid")
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = fftconvolve(img1 * img1, window, mode="valid") - mu1_sq
    sigma2_sq = fftconvolve(img2 * img2, window, mode="valid") - mu2_sq
    sigma12 = fftconvolve(img1 * img2, window, mode="valid") - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def masked_ssim(orig, restored, mask2d, margin=60, eps=1e-8):
    """
    손상 영역(bounding box + margin, zoom 시각화와 동일한 범위)을 크롭해서
    11x11 가우시안 윈도우 기반 SSIM 계산. 1에 가까울수록 좋음.
    데이터가 부족하면(크롭이 11px보다 작으면) NaN 반환.
    """
    h, w = mask2d.shape
    damaged_rows = np.where(mask2d.any(axis=1))[0]
    if len(damaged_rows) == 0:
        return np.nan
    r0 = max(0, damaged_rows.min() - margin)
    r1 = min(h, damaged_rows.max() + margin)
    damaged_cols = np.where(mask2d.any(axis=0))[0]
    c0 = max(0, damaged_cols.min() - 20)
    c1 = min(w, damaged_cols.max() + 20)

    crop_orig = orig[r0:r1, c0:c1].astype(np.float64)
    crop_restored = restored[r0:r1, c0:c1].astype(np.float64)

    if crop_orig.shape[0] < 11 or crop_orig.shape[1] < 11:
        return np.nan

    ref_min, ref_max = crop_orig.min(), crop_orig.max()
    denom = max(ref_max - ref_min, eps)
    crop_orig_n = (crop_orig - ref_min) / denom
    crop_restored_n = (crop_restored - ref_min) / denom

    window = _gaussian_window(11, 1.5)
    ssim_map = _ssim_map(crop_orig_n, crop_restored_n, window, data_range=1.0)
    return float(ssim_map.mean())

def restore_vertical_linear(damaged, mask2d):
    """
    열(column) 단위 1D 선형 보간. cubic과 달리 두 경계값을 직선으로만 잇기 때문에
    구간이 아무리 넓어도 보간값이 두 경계값 사이 범위를 벗어날 수 없음
    (구조적으로 overshoot 불가능).

    damaged: (H, W) 손상된 이미지 배열
    mask2d : (H, W) {0,1}, 1=손상(복원 대상)
    반환   : (H, W) 복원된 배열 (float32)
    """
    h, w = damaged.shape
    out = damaged.copy().astype(np.float32)
    rows = np.arange(h)

    damaged_cols = np.where(mask2d.any(axis=0))[0]

    for col in damaged_cols:
        col_mask = mask2d[:, col] > 0
        damaged_row_idx = rows[col_mask]
        if len(damaged_row_idx) == 0:
            continue

        known_rows = rows[~col_mask]
        known_vals = damaged[~col_mask, col]

        if len(known_rows) < 2:
            continue

        interp_vals = np.interp(damaged_row_idx, known_rows, known_vals)
        out[damaged_row_idx, col] = interp_vals

    return out