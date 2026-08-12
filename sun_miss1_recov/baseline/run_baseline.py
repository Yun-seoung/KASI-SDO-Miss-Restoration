"""
run_baseline.py — MISS1 단계 baseline: normal 이미지 + MISS1 마스크뱅크
-> 세로 방향 보간(cubic spline 또는 linear) 복원 -> 오차 계산

normal 이미지 목록은 sun_miss0_recov/data/의 것을 그대로 재사용함
(MISS 종류와 무관하게 같은 정상 이미지 풀, val 200장도 동일하게 재사용).

[통일] 시각화를 나머지 7개(AE/VAE/LaMa/Palette × MISS0/MISS1)와 동일한
스타일로 통일: 컬러바 삭제, min-max([0,1]) 정규화된 Abs error, inferno 컬러맵.

[수정 2026-07] cubic spline이 넓은 결측 구간에서 overshoot(경계 조건상
비정상적으로 크거나 작은 값을 뱉는 현상)을 일으켜 시각화가 흑백으로 clipping되고
지표가 과장되게 나쁘게 나오는 문제를 확인. 대응:
  1) 복원값을 원본 이미지의 실제 값 범위로 clip (물리적으로 타당한 범위 강제)
  2) --method linear 옵션 추가 — 구조적으로 overshoot이 없는 선형 보간과
     cubic을 나란히 비교할 수 있게 함

"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # sun_miss1_recov/
SIBLING_ROOT = os.path.dirname(ROOT_DIR)  # cnu_ck_sdo_recov/
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank
from restore_cubic import (
    restore_vertical_cubic, apply_damage, masked_l1, masked_rmse,
    masked_psnr, masked_region_stats, masked_ssim,
)

VAL_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_val_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss1.npy")
VERIFY_DIR = os.path.join(ROOT_DIR, "verify_out")
os.makedirs(VERIFY_DIR, exist_ok=True)


def get_results_path(method):
    return os.path.join(ROOT_DIR, "results", f"baseline_results_{method}.csv")


def restore_vertical_linear(damaged, mask2d):
    """
    열(column) 단위 1D 선형 보간. cubic과 달리 두 경계값을 직선으로만 잇기 때문에
    구간이 아무리 넓어도 보간값이 두 경계값 사이 범위를 벗어날 수 없음
    (구조적으로 overshoot 불가능).

    damaged: (H, W) 배열, 손상 위치는 fill_value(보통 0.0)로 채워져 있음
    mask2d : (H, W) bool 배열, True = 손상 위치
    """
    restored = damaged.copy().astype(np.float32)
    h, w = damaged.shape
    rows = np.arange(h)

    for c in range(w):
        col_mask = mask2d[:, c]
        if not col_mask.any():
            continue
        known_rows = rows[~col_mask]
        if len(known_rows) < 2:
            # 참고할 정상 픽셀이 1개 이하면 보간 불가 -> 그대로 둠(혹은 최근접값)
            if len(known_rows) == 1:
                restored[col_mask, c] = damaged[known_rows[0], c]
            continue
        known_vals = damaged[known_rows, c]
        missing_rows = rows[col_mask]
        restored[missing_rows, c] = np.interp(missing_rows, known_rows, known_vals)

    return restored


def get_crop_bounds(mask2d, margin=60):
    h, w = mask2d.shape
    damaged_rows = np.where(mask2d.any(axis=1))[0]
    if len(damaged_rows) == 0:
        return 0, h, 0, w
    r0 = max(0, damaged_rows.min() - margin)
    r1 = min(h, damaged_rows.max() + margin)
    damaged_cols = np.where(mask2d.any(axis=0))[0]
    c0 = max(0, damaged_cols.min() - 20)
    c1 = min(w, damaged_cols.max() + 20)
    return r0, r1, c0, c1


def masked_relative_l1_global(orig, l1, precomputed_scale=None, eps=1e-6):
    scale = precomputed_scale
    if scale is None:
        scale = np.percentile(orig[orig > 0], 99) if (orig > 0).any() else np.nan
    if scale is None or np.isnan(scale) or scale < eps:
        return np.nan
    return float(l1 / scale * 100.0)


def visualize(orig, damaged, restored, mask2d, out_path, vmax):
    """
    [통일] AE/VAE/LaMa/Palette와 동일한 시각화 스타일:
    컬러바 없음, min-max([0,1]) 정규화된 Abs error, inferno 컬러맵,
    마스킹 영역 기준 percentile 95 대비 강조.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp_orig = np.nan_to_num(orig, nan=0.0)

    # [min-max 정규화] masked_psnr과 동일 기준(이미지 전체 min/max)으로 오차 계산
    o_min, o_max = float(orig.min()), float(orig.max())
    denom = max(o_max - o_min, 1e-8)
    orig_n = (disp_orig.astype(np.float32) - o_min) / denom
    restored_n = (restored.astype(np.float32) - o_min) / denom
    diff = np.abs(orig_n - restored_n)

    r0, r1, c0, c1 = get_crop_bounds(mask2d)
    l1 = masked_l1(orig, restored, mask2d)
    rel_l1_global = masked_relative_l1_global(orig, l1, precomputed_scale=vmax)
    psnr = masked_psnr(orig, restored, mask2d)

    zoom_diff = diff[r0:r1, c0:c1]
    zoom_mask = mask2d[r0:r1, c0:c1] > 0
    diff_vmax_zoom = max(np.percentile(zoom_diff[zoom_mask], 95), 1e-6) if zoom_mask.any() \
        else max(np.percentile(zoom_diff, 95), 1e-6)
    diff_vmax_full = max(np.percentile(diff, 99.5), 1e-6)

    def downsample(a, factor=8):
        return a[::factor, ::factor]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    titles_top = ["Original (full, downsampled)", "Damaged (full, downsampled)",
                  "Restored (full, downsampled)", "Abs error (full, downsampled)"]
    imgs_top = [downsample(disp_orig), downsample(damaged),
                downsample(restored), downsample(diff)]
    cmaps_top = ["gray", "gray", "gray", "inferno"]
    ds_factor = 8
    for ax, im, title, cmap in zip(axes[0], imgs_top, titles_top, cmaps_top):
        if cmap == "gray":
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=diff_vmax_full)
        ax.set_title(title, fontsize=10)
        ax.add_patch(plt.Rectangle((c0 / ds_factor, r0 / ds_factor),
                                    (c1 - c0) / ds_factor, (r1 - r0) / ds_factor,
                                    fill=False, edgecolor="lime", linewidth=1.2))
        ax.axis("off")

    titles_bot = ["Original (zoom)", "Damaged (zoom)", "Restored (zoom)", "Abs error (zoom)"]
    imgs_bot = [disp_orig[r0:r1, c0:c1], damaged[r0:r1, c0:c1],
                restored[r0:r1, c0:c1], zoom_diff]
    for ax, im, title, cmap in zip(axes[1], imgs_bot, titles_bot, cmaps_top):
        if cmap == "gray":
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=diff_vmax_zoom, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"mask ratio={mask2d.mean():.4f} | masked_l1={l1:.2f} | "
        f"relative_l1(global scale)={rel_l1_global:.2f}% | PSNR={psnr:.2f}dB",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def main(method="cubic", n_images=200, n_masks_per_image=1, save_viz_n=5, seed=0):
    if not os.path.exists(VAL_CSV):
        print(f"[중단] val 이미지 리스트 없음: {VAL_CSV}")
        return
    if not os.path.exists(MASK_BANK_PATH):
        print(f"[중단] 마스크뱅크 없음: {MASK_BANK_PATH}")
        return

    results_path = get_results_path(method)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    val_df = pd.read_csv(VAL_CSV)
    if len(val_df) < n_images:
        n_images = len(val_df)
    val_df = val_df.sample(n=n_images, random_state=seed).reset_index(drop=True)

    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=seed)

    results = []
    t0 = time.time()
    total_clipped_pixels = 0

    for i, row in val_df.iterrows():
        img_t0 = time.time()
        img = np.load(row["cache_path"])
        t_load = time.time() - img_t0

        if np.isnan(img).any():
            print(f"[warn] normal인데 NaN 존재 -> 스킵: {row['path']}")
            continue

        vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0

        for k in range(n_masks_per_image):
            mask2d = mask_bank.sample()
            damaged = apply_damage(img, mask2d, fill_value=0.0)

            t_restore0 = time.time()
            if method == "cubic":
                restored = restore_vertical_cubic(damaged, mask2d)
            elif method == "linear":
                restored = restore_vertical_linear(damaged, mask2d)
            else:
                raise ValueError(f"알 수 없는 method: {method}")

            # overshoot 방지: 복원값을 원본 이미지의 실제 값 범위로 clip
            # (이 원본 img 자체가 정상 이미지이므로, img.min()/img.max()가
            #  물리적으로 타당한 픽셀값 범위)
            before_clip = restored.copy()
            restored = np.clip(restored, float(img.min()), float(img.max()))
            n_clipped = int(np.sum(before_clip != restored))
            total_clipped_pixels += n_clipped

            t_restore = time.time() - t_restore0

            l1 = masked_l1(img, restored, mask2d)
            rmse = masked_rmse(img, restored, mask2d)
            psnr_val = masked_psnr(img, restored, mask2d)
            ssim_val = masked_ssim(img, restored, mask2d)
            rel_l1_global = masked_relative_l1_global(img, l1, precomputed_scale=vmax)
            region_stats = masked_region_stats(img, mask2d)

            t_viz = 0.0
            if len(results) < save_viz_n:
                fname = os.path.splitext(os.path.basename(row["path"]))[0]
                out_path = os.path.join(VERIFY_DIR, f"baseline_{method}_{fname}_m{k}.png")
                t_viz0 = time.time()
                visualize(img, damaged, restored, mask2d, out_path, vmax)
                t_viz = time.time() - t_viz0
                print(f"    시각화 저장 → {out_path}")

            results.append({
                "path": row["path"],
                "mask_idx": k,
                "mask_ratio": float(mask2d.mean()),
                "masked_l1": l1,
                "masked_rmse": rmse,
                "masked_psnr": psnr_val, "masked_ssim": ssim_val,
                "relative_l1_global_pct": rel_l1_global,
                "region_mean": region_stats["mean"],
                "region_std": region_stats["std"],
                "n_clipped_pixels": n_clipped,
                "load_seconds": t_load,
                "restore_seconds": t_restore,
                "viz_seconds": t_viz,
            })

            print(f"    [{i+1}/{n_images}] method={method} load={t_load:.2f}s restore={t_restore:.2f}s "
                  f"viz={t_viz:.2f}s | masked_l1={l1:.2f} rel_l1={rel_l1_global:.2f}% "
                  f"PSNR={psnr_val:.2f}dB clipped={n_clipped}px")

    if not results:
        print("[중단] 결과 없음")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(results_path, index=False)

    print(f"\n[완료] MISS1 baseline({method}) 결과 {len(res_df)}건 저장 → {results_path}")
    print(f"  평균 masked_l1              = {res_df['masked_l1'].mean():.4f}")
    print(f"  평균 masked_rmse            = {res_df['masked_rmse'].mean():.4f}")
    finite_psnr = res_df['masked_psnr'].replace([np.inf, -np.inf], np.nan)
    print(f"  평균 masked_psnr(dB)        = {finite_psnr.mean():.2f}")
    print(f"  평균 masked_ssim      = {res_df['masked_ssim'].mean():.4f}")
    print(f"  평균 relative_l1(global, %) = {res_df['relative_l1_global_pct'].mean():.2f}%")
    print(f"  clip된 픽셀 총합            = {total_clipped_pixels}")
    print(f"  전체 wall-clock 경과 = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, choices=["cubic", "linear"], default="cubic",
                     help="복원 방법: cubic(기존 spline) 또는 linear(신규 선형 보간)")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--n_masks_per_image", type=int, default=1)
    ap.add_argument("--save_viz_n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(method=args.method, n_images=args.n_images, n_masks_per_image=args.n_masks_per_image,
         save_viz_n=args.save_viz_n, seed=args.seed)