"""
evaluate_lama.py — LaMa Generator로 val 200장을 복원하고 baseline과 동일 지표로 평가.

[알고리즘 1] 학습 patch 생성 시 손상 덩어리 단위 crop 적용됨(precompute_patches.py)
[알고리즘 2] 겹치는 타일 병합 전략 선택 가능: --merge_mode average|best_tile
  (patch_geometry.merge_tiles 사용, 패치 중심 기반 신뢰도로 가중평균 또는 경성선택)
[SSIM] masked_ssim 지표 포함

evaluate_ckpt.py와의 차이:
  - normal_paths_val_cached.csv + mask_bank_miss0.npy 사용 (baseline/AE/VAE와 동일 val 셋)
  - 평가 노이즈는 sun_lama.py의 make_eval_noise()로 (index, seed) 기반 결정론적 생성
  - Discriminator는 평가에 불필요하므로 Generator만 로드
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, os.path.join(ROOT_DIR, "baseline"))
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank
from patch_geometry import get_damage_components, _gen_starts, merge_tiles
from sun_lama import build_generator, make_lama_input, make_eval_noise
from restore_cubic import masked_l1, masked_rmse, masked_psnr, masked_region_stats, masked_ssim

VAL_CSV = os.path.join(ROOT_DIR, "data", "normal_paths_val_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss0.npy")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "lama_results.csv")
VERIFY_DIR = os.path.join(ROOT_DIR, "verify_out")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

PATCH_SIZE = 256
STRIDE = 64  # precompute_patches.py와 통일 (기존 128에서 축소)
EVAL_NOISE_STD = 1.0


def get_crop_bounds_full(mask2d, margin=60):
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


def normalize_01(img2d, eps=1e-8):
    ref_min = float(img2d.min())
    ref_max = float(img2d.max())
    denom = max(ref_max - ref_min, eps)
    return (img2d.astype(np.float32) - ref_min) / denom, ref_min, ref_max


def denormalize(img_n, ref_min, ref_max):
    return img_n * (ref_max - ref_min) + ref_min


def restore_with_lama(G, img_full, mask_full, device, index, seed, merge_mode="average",
                       patch_size=PATCH_SIZE, stride=STRIDE, noise_std=EVAL_NOISE_STD):
    """
    [알고리즘 1] 손상 덩어리 단위로 crop 위치 선정.
    [알고리즘 2] merge_mode에 따라 겹치는 타일을 가중평균 또는 경성선택으로 병합.
    """
    h, w = img_full.shape
    boxes = get_damage_components(mask_full)
    if not boxes:
        return img_full.copy(), 1.0

    seen = set()
    tile_results = []
    tile_idx = 0
    G.eval()
    with torch.no_grad():
        for (r0, r1, c0, c1) in boxes:
            row_starts = _gen_starts(r0, r1, patch_size, stride, h)
            col_starts = _gen_starts(c0, c1, patch_size, stride, w)
            for rs in row_starts:
                for cs in col_starts:
                    key = (rs, cs)
                    if key in seen:
                        continue
                    mask_patch = mask_full[rs:rs + patch_size, cs:cs + patch_size]
                    if mask_patch.sum() == 0:
                        continue
                    seen.add(key)
                    img_patch = img_full[rs:rs + patch_size, cs:cs + patch_size]

                    img_n, ref_min, ref_max = normalize_01(img_patch)
                    img_t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).float().to(device)
                    mask_t = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    # 결정론적 평가 노이즈: (index, tile_idx, seed) 조합으로 고정
                    noise = make_eval_noise((1, patch_size, patch_size),
                                             index=index * 1000 + tile_idx, seed=seed,
                                             noise_std=noise_std).unsqueeze(0).to(device)
                    tile_idx += 1

                    pred_n = G(make_lama_input(img_t, mask_t, noise=noise))
                    pred_n = pred_n.squeeze(0).squeeze(0).cpu().numpy()
                    pred_denorm = denormalize(pred_n, ref_min, ref_max)

                    tile_results.append((rs, cs, pred_denorm, mask_patch))

    if not tile_results:
        return img_full.copy(), 1.0

    merged, covered = merge_tiles(h, w, tile_results, merge_mode=merge_mode)

    restored_full = img_full.copy()
    restored_full[covered] = merged[covered]

    total_damaged = (mask_full > 0).sum()
    coverage = float(covered[mask_full > 0].sum()) / total_damaged if total_damaged > 0 else 1.0

    return restored_full, coverage


def visualize(orig, damaged, restored, mask2d, out_path, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # [min-max 정규화] masked_psnr과 동일 기준(이미지 전체 min/max)으로 오차 계산
    o_min, o_max = float(orig.min()), float(orig.max())
    denom = max(o_max - o_min, 1e-8)
    orig_n = (orig.astype(np.float32) - o_min) / denom
    restored_n = (restored.astype(np.float32) - o_min) / denom
    diff = np.abs(orig_n - restored_n)
    r0f, r1f, c0f, c1f = get_crop_bounds_full(mask2d)

    zoom_diff = diff[r0f:r1f, c0f:c1f]
    zoom_mask = mask2d[r0f:r1f, c0f:c1f] > 0
    diff_vmax_zoom = max(np.percentile(zoom_diff[zoom_mask], 95), 1e-6) if zoom_mask.any() \
        else max(np.percentile(zoom_diff, 95), 1e-6)
    diff_vmax_full = max(np.percentile(diff, 99.5), 1e-6)

    def downsample(a, factor=8):
        return a[::factor, ::factor]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    titles_top = ["Original (full, downsampled)", "Damaged (full, downsampled)",
                  "LaMa Restored (full, downsampled)", "Abs error (full, downsampled)"]
    imgs_top = [downsample(orig), downsample(damaged), downsample(restored), downsample(diff)]
    cmaps_top = ["gray", "gray", "gray", "inferno"]
    ds_factor = 8
    for ax, im, title, cmap in zip(axes[0], imgs_top, titles_top, cmaps_top):
        if cmap == "gray":
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=diff_vmax_full)
        ax.set_title(title, fontsize=10)
        ax.add_patch(plt.Rectangle((c0f / ds_factor, r0f / ds_factor),
                                    (c1f - c0f) / ds_factor, (r1f - r0f) / ds_factor,
                                    fill=False, edgecolor="lime", linewidth=1.2))
        ax.axis("off")

    titles_bot = ["Original (zoom)", "Damaged (zoom)", "LaMa Restored (zoom)", "Abs error (zoom)"]
    imgs_bot = [orig[r0f:r1f, c0f:c1f], damaged[r0f:r1f, c0f:c1f],
                restored[r0f:r1f, c0f:c1f], zoom_diff]
    for ax, im, title, cmap in zip(axes[1], imgs_bot, titles_bot, cmaps_top):
        if cmap == "gray":
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=diff_vmax_zoom, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    l1 = masked_l1(orig, restored, mask2d)
    psnr_val = masked_psnr(orig, restored, mask2d)
    fig.suptitle(f"mask ratio={mask2d.mean():.4f} | masked_l1={l1:.2f} | PSNR={psnr_val:.2f}dB",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def main(ckpt_path, n_images=200, save_viz_n=5, seed=0, merge_mode="average"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, merge_mode={merge_mode}")

    G = build_generator(img_ch=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    print(f"[체크포인트 로드] {ckpt_path} (epoch {ckpt['epoch']})")

    val_df = pd.read_csv(VAL_CSV)
    if len(val_df) < n_images:
        n_images = len(val_df)
    val_df = val_df.sample(n=n_images, random_state=seed).reset_index(drop=True)

    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=seed)

    results = []
    t0 = time.time()

    for i, row in val_df.iterrows():
        img = np.load(row["cache_path"])
        vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0

        mask2d = mask_bank.sample()
        damaged = img.copy()
        damaged[mask2d > 0] = 0.0

        restored, coverage = restore_with_lama(G, img, mask2d, device, index=i, seed=seed,
                                                merge_mode=merge_mode)

        l1 = masked_l1(img, restored, mask2d)
        rmse = masked_rmse(img, restored, mask2d)
        psnr_val = masked_psnr(img, restored, mask2d)
        ssim_val = masked_ssim(img, restored, mask2d)
        region_stats = masked_region_stats(img, mask2d)

        if i < save_viz_n:
            fname = os.path.splitext(os.path.basename(row["path"]))[0]
            out_path = os.path.join(VERIFY_DIR, f"lama_{fname}_m0.png")
            visualize(img, damaged, restored, mask2d, out_path, vmax)
            print(f"    시각화 저장 → {out_path}")

        results.append({
            "path": row["path"], "mask_ratio": float(mask2d.mean()),
            "coverage": coverage, "masked_l1": l1, "masked_rmse": rmse,
            "masked_psnr": psnr_val, "masked_ssim": ssim_val,
            "region_mean": region_stats["mean"],
        })

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{n_images} (직전 coverage={coverage:.3f})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n[완료] LaMa 평가 결과 {len(res_df)}건 저장 → {RESULTS_PATH}")
    print(f"  merge_mode = {merge_mode}")
    print(f"  평균 coverage    = {res_df['coverage'].mean():.4f}")
    print(f"  평균 masked_l1   = {res_df['masked_l1'].mean():.4f}")
    print(f"  평균 masked_rmse = {res_df['masked_rmse'].mean():.4f}")
    finite_psnr = res_df['masked_psnr'].replace([np.inf, -np.inf], np.nan)
    print(f"  평균 masked_psnr(dB) = {finite_psnr.mean():.2f}")
    print(f"  평균 masked_ssim      = {res_df['masked_ssim'].mean():.4f}")
    print(f"  전체 경과 = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--save_viz_n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge_mode", type=str, default="average", choices=["average", "best_tile"])
    args = ap.parse_args()
    main(ckpt_path=args.ckpt, n_images=args.n_images,
         save_viz_n=args.save_viz_n, seed=args.seed, merge_mode=args.merge_mode)