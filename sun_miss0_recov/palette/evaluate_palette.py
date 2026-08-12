"""
evaluate_palette.py — 원본 Palette Network로 MISS0 val 200장을 복원하고
baseline/AE/VAE/LaMa와 동일한 지표로 평가. 마스크는 MISS0 단계 마스크뱅크 사용.

원본 net.restoration(y_cond, y_0=, mask=, sample_num=)이 reverse diffusion을
수행하며 매 스텝 "y_t = y_0*(1-mask) + mask*y_t"를 적용(내부 로직).

참고: MISS0은 손상 비율이 좁아(0.28%) MISS1보다 손상 bounding box당
      타일 개수가 적음 -> 동일 설정 기준 MISS1보다는 빠를 것으로 예상.
      --sample_steps로 속도 조절 (기본 50).

[수정 2026-07] ssim_val 라인 IndentationError 수정.
[수정 2026-07] --sample_steps가 실제로 반영되지 않던 버그 수정
  (set_eval_noise_schedule에 n_timestep 파라미터 추가, BETA_SCHEDULE["test"] 덮어씀).
[수정 2026-07] STRIDE 128(50% 겹침) -> 64(75% 겹침)로 변경, 다른 방법(LaMa 등)과 통일.
[수정 2026-07] MASK_BANK_PATH를 mask_bank_miss0.npy로 수정 (MISS1용 경로가 하드코딩되어 있던 버그).

"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # sun_miss0_recov/
SIBLING_ROOT = os.path.dirname(ROOT_DIR)  # cnu_ck_sdo_recov/
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")
MISS0_BASELINE_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "baseline")

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, MISS0_BASELINE_DIR)  # restore_cubic.py 재사용
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank
from sun_palette import build_palette_network, set_eval_noise_schedule
from restore_cubic import masked_l1, masked_rmse, masked_psnr, masked_region_stats, masked_ssim

VAL_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_val_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss0.npy")  # [수정] miss1 -> miss0
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "palette_results.csv")
VERIFY_DIR = os.path.join(ROOT_DIR, "verify_out")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

PATCH_SIZE = 256
STRIDE = 64


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


def _gen_starts(lo, hi, size, stride, max_val):
    if hi - lo <= size:
        start = int(np.clip((lo + hi) // 2 - size // 2, 0, max_val - size))
        return [start]
    starts = list(range(lo, hi - size + 1, stride))
    if not starts or starts[-1] != hi - size:
        starts.append(hi - size)
    return [int(np.clip(s, 0, max_val - size)) for s in starts]


def restore_with_palette(net, img_full, mask_full, device, sample_num=1,
                          patch_size=PATCH_SIZE, stride=STRIDE):
    h, w = img_full.shape
    damaged_rows = np.where(mask_full.any(axis=1))[0]
    if len(damaged_rows) == 0:
        return img_full.copy(), 1.0

    r0b, r1b = int(damaged_rows.min()), int(damaged_rows.max()) + 1
    damaged_cols = np.where(mask_full.any(axis=0))[0]
    c0b, c1b = int(damaged_cols.min()), int(damaged_cols.max()) + 1

    row_starts = _gen_starts(r0b, r1b, patch_size, stride, h)
    col_starts = _gen_starts(c0b, c1b, patch_size, stride, w)

    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    net.eval()
    for rs in row_starts:
        for cs in col_starts:
            mask_patch = mask_full[rs:rs + patch_size, cs:cs + patch_size]
            if mask_patch.sum() == 0:
                continue
            img_patch = img_full[rs:rs + patch_size, cs:cs + patch_size]

            img_n, ref_min, ref_max = normalize_01(img_patch)
            mask_t = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0).float().to(device)
            y_0 = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).float().to(device)

            noise = torch.randn_like(y_0) * 1.0
            y_cond = torch.where(mask_t.bool(), noise, y_0)

            y_t, _ = net.restoration(y_cond, y_0=y_0, mask=mask_t, sample_num=sample_num)

            pred_n = y_t.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_denorm = denormalize(pred_n, ref_min, ref_max)

            accum[rs:rs + patch_size, cs:cs + patch_size] += pred_denorm * mask_patch
            weight[rs:rs + patch_size, cs:cs + patch_size] += mask_patch

    restored_full = img_full.copy()
    covered = weight > 0
    restored_full[covered] = accum[covered] / weight[covered]

    total_damaged = (mask_full > 0).sum()
    coverage = float(covered[mask_full > 0].sum()) / total_damaged if total_damaged > 0 else 1.0

    return restored_full, coverage


def visualize(orig, damaged, restored, mask2d, out_path, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
                  "Palette Restored (full, downsampled)", "Abs error (full, downsampled)"]
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

    titles_bot = ["Original (zoom)", "Damaged (zoom)", "Palette Restored (zoom)", "Abs error (zoom)"]
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


def main(ckpt_path, n_images=50, save_viz_n=5, seed=0, sample_steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, sample_steps={sample_steps}, stride={STRIDE}")

    net = build_palette_network(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["net"])
    set_eval_noise_schedule(net, device=device, n_timestep=sample_steps)
    print(f"[체크포인트 로드] {ckpt_path} (epoch {ckpt['epoch']+1})")

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

        restored, coverage = restore_with_palette(net, img, mask2d, device)

        l1 = masked_l1(img, restored, mask2d)
        rmse = masked_rmse(img, restored, mask2d)
        psnr_val = masked_psnr(img, restored, mask2d)
        ssim_val = masked_ssim(img, restored, mask2d)
        region_stats = masked_region_stats(img, mask2d)

        if i < save_viz_n:
            fname = os.path.splitext(os.path.basename(row["path"]))[0]
            out_path = os.path.join(VERIFY_DIR, f"palette_{fname}_m0.png")
            visualize(img, damaged, restored, mask2d, out_path, vmax)
            print(f"    시각화 저장 → {out_path}")

        results.append({
            "path": row["path"], "mask_ratio": float(mask2d.mean()),
            "coverage": coverage, "masked_l1": l1, "masked_rmse": rmse,
            "masked_psnr": psnr_val, "masked_ssim": ssim_val, "region_mean": region_stats["mean"],
        })

        elapsed_so_far = time.time() - t0
        print(f"  [{i+1}/{n_images}] coverage={coverage:.3f} masked_l1={l1:.2f} "
              f"psnr={psnr_val:.2f}dB (누적 {elapsed_so_far:.0f}s)")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n[완료] Palette(MISS0) 평가 결과 {len(res_df)}건 저장 → {RESULTS_PATH}")
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
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--save_viz_n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_steps", type=int, default=50)
    args = ap.parse_args()
    main(ckpt_path=args.ckpt, n_images=args.n_images,
         save_viz_n=args.save_viz_n, seed=args.seed, sample_steps=args.sample_steps)