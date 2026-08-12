"""
evaluate_palette.py — 원본 Palette Network로 MISS1 val 200장을 복원하고
baseline/AE/VAE/LaMa와 동일한 지표로 평가. 마스크는 MISS1 단계 마스크뱅크 사용.

원본 net.restoration(y_cond, y_0=, mask=, sample_num=)이 reverse diffusion을
수행하며 매 스텝 "y_t = y_0*(1-mask) + mask*y_t"를 적용(내부 로직).

주의: MISS1은 손상 비율이 넓어(2.66%) 손상 bounding box당 타일 개수가
      MISS0보다 많음 -> patch당 200-step 샘플링과 겹쳐 MISS0보다도 느릴 수 있음.
      --sample_steps로 속도 조절 (기본 50).

[알고리즘 1] 손상 덩어리 단위 crop 적용 (get_damage_components, mask_bank/patch_geometry.py 공용 모듈)
[알고리즘 2] 겹치는 타일 병합 전략 선택 가능: --merge_mode average|best_tile|core_crop
[core_crop] AE/VAE/LaMa와 동일 방식 이식. 패치 크기 L=512, stride=L/2(=256)로 추론하고,
            각 패치의 중심부 L/2(=256)만 남기고 가장자리 L/4(=128)씩 잘라낸 뒤 이어붙임.
[core grid] core_crop 모드는 박스별 grid 대신 이미지 전체 기준 원점(0)
            고정 전역 grid(_core_grid_starts) 사용 — AE/VAE/LaMa와 동일.
[seam correction 미적용] AE/VAE/LaMa와 동일하게 순정 core_crop만 적용.

[수정 2026-07] ssim_val 라인 IndentationError 수정.
[수정 2026-07] --sample_steps가 실제로 반영되지 않던 버그 수정
  (set_eval_noise_schedule에 n_timestep 파라미터 추가, BETA_SCHEDULE["test"] 덮어씀).
[수정 2026-07] STRIDE 128(50% 겹침) -> 64(75% 겹침)로 변경, 다른 방법(LaMa 등)과 통일.
[수정 2026-08-10] AE/VAE/LaMa와 동일한 core_crop 병합 로직, fallback 픽셀 카운트,
  patch_geometry 공용 모듈(get_damage_components/_gen_starts/merge_tiles) 이식.
  covered를 mask_full>0과 교집합하도록 수정 (블록 아티팩트 버그 방지).
[수정 2026-08-10] 장시간(2일+) 실행 대비 SAVE_EVERY장마다 중간 CSV 저장 추가.
  루프 중간에 프로세스가 죽어도 마지막 저장 시점까지 결과 보존.

"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # sun_miss1_recov/
SIBLING_ROOT = os.path.dirname(ROOT_DIR)  # cnu_ck_sdo_recov/
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")
MISS0_BASELINE_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "baseline")

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, MISS0_BASELINE_DIR)  # restore_cubic.py 재사용
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank
from patch_geometry import get_damage_components, _gen_starts, merge_tiles
from sun_palette import build_palette_network, set_eval_noise_schedule
from restore_cubic import masked_l1, masked_rmse, masked_psnr, masked_region_stats, masked_ssim

VAL_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_val_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss1.npy")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "palette_results.csv")
VERIFY_DIR = os.path.join(ROOT_DIR, "verify_out")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

PATCH_SIZE = 512
STRIDE = 128  # average / best_tile 모드용 (AE/VAE/LaMa와 동일)

# --- core_crop 모드 전용 파라미터 (AE/VAE/LaMa와 동일) ---
CORE_MARGIN = PATCH_SIZE // 4   # 128
CORE_STRIDE = PATCH_SIZE // 2   # 256

# --- 장시간 실행 대비 중간 저장 주기 ---
SAVE_EVERY = 20  # 이 장수마다 results CSV를 중간 저장 (덮어쓰기)


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


def _core_grid_starts(total, patch_size, stride):
    if total <= patch_size:
        return [0]
    starts = list(range(0, total - patch_size + 1, stride))
    if starts[-1] != total - patch_size:
        starts.append(total - patch_size)
    return starts


def core_crop_merge(h, w, tile_results, patch_size, core_margin):
    merged = np.zeros((h, w), dtype=np.float32)
    covered = np.zeros((h, w), dtype=bool)

    core = core_margin
    for (rs, cs, pred_denorm, mask_patch) in tile_results:
        r0, r1 = rs + core, rs + patch_size - core
        c0, c1 = cs + core, cs + patch_size - core
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, h), min(c1, w)
        if r1 <= r0 or c1 <= c0:
            continue

        pr0, pr1 = r0 - rs, r1 - rs
        pc0, pc1 = c0 - cs, c1 - cs

        merged[r0:r1, c0:c1] = pred_denorm[pr0:pr1, pc0:pc1]
        covered[r0:r1, c0:c1] = True

    fallback_sum = np.zeros((h, w), dtype=np.float32)
    fallback_cnt = np.zeros((h, w), dtype=np.float32)
    for (rs, cs, pred_denorm, mask_patch) in tile_results:
        r0, r1 = rs, rs + patch_size
        c0, c1 = cs, cs + patch_size
        region_uncovered = ~covered[r0:r1, c0:c1]
        if region_uncovered.any():
            fallback_sum[r0:r1, c0:c1][region_uncovered] += pred_denorm[region_uncovered]
            fallback_cnt[r0:r1, c0:c1][region_uncovered] += 1.0

    fill_mask = (fallback_cnt > 0) & (~covered)
    if fill_mask.any():
        merged[fill_mask] = fallback_sum[fill_mask] / fallback_cnt[fill_mask]
        covered[fill_mask] = True

    return merged, covered, fill_mask


def restore_with_palette(net, img_full, mask_full, device, merge_mode="average",
                          patch_size=PATCH_SIZE, stride=None, sample_num=1):
    if stride is None:
        stride = CORE_STRIDE if merge_mode == "core_crop" else STRIDE

    h, w = img_full.shape
    boxes = get_damage_components(mask_full)
    if not boxes:
        return img_full.copy(), 1.0, 0

    seen = set()
    tile_results = []
    net.eval()

    if merge_mode == "core_crop":
        row_starts_all = _core_grid_starts(h, patch_size, stride)
        col_starts_all = _core_grid_starts(w, patch_size, stride)
        candidates = [(rs, cs) for rs in row_starts_all for cs in col_starts_all]
    else:
        candidates = []
        for (r0, r1, c0, c1) in boxes:
            row_starts = _gen_starts(r0, r1, patch_size, stride, h)
            col_starts = _gen_starts(c0, c1, patch_size, stride, w)
            candidates.extend((rs, cs) for rs in row_starts for cs in col_starts)

    with torch.no_grad():
        for rs, cs in candidates:
            key = (rs, cs)
            if key in seen:
                continue
            mask_patch = mask_full[rs:rs + patch_size, cs:cs + patch_size]
            if mask_patch.sum() == 0:
                continue
            seen.add(key)
            img_patch = img_full[rs:rs + patch_size, cs:cs + patch_size]

            img_n, ref_min, ref_max = normalize_01(img_patch)
            mask_t = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0).float().to(device)
            y_0 = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).float().to(device)

            noise = torch.randn_like(y_0) * 1.0
            y_cond = torch.where(mask_t.bool(), noise, y_0)

            y_t, _ = net.restoration(y_cond, y_0=y_0, mask=mask_t, sample_num=sample_num)

            pred_n = y_t.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_denorm = denormalize(pred_n, ref_min, ref_max)

            tile_results.append((rs, cs, pred_denorm, mask_patch))

    if not tile_results:
        return img_full.copy(), 1.0, 0

    fallback_pixel_count = 0
    if merge_mode == "core_crop":
        merged, covered, fallback_mask = core_crop_merge(h, w, tile_results, patch_size, CORE_MARGIN)
        fallback_pixel_count = int((fallback_mask & (mask_full > 0)).sum())
    else:
        merged, covered = merge_tiles(h, w, tile_results, merge_mode=merge_mode)

    restored_full = img_full.copy()
    final_mask = covered & (mask_full > 0)
    restored_full[final_mask] = merged[final_mask]

    total_damaged = (mask_full > 0).sum()
    coverage = float(covered[mask_full > 0].sum()) / total_damaged if total_damaged > 0 else 1.0

    return restored_full, coverage, fallback_pixel_count


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


def main(ckpt_path, n_images=200, save_viz_n=5, seed=0, sample_steps=50, merge_mode="average"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, sample_steps={sample_steps}, merge_mode={merge_mode}")

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
    total_fallback_pixels = 0

    for i, row in val_df.iterrows():
        img = np.load(row["cache_path"])
        vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0

        mask2d = mask_bank.sample()
        damaged = img.copy()
        damaged[mask2d > 0] = 0.0

        restored, coverage, fallback_px = restore_with_palette(
            net, img, mask2d, device, merge_mode=merge_mode)
        total_fallback_pixels += fallback_px

        l1 = masked_l1(img, restored, mask2d)
        rmse = masked_rmse(img, restored, mask2d)
        psnr_val = masked_psnr(img, restored, mask2d)
        ssim_val = masked_ssim(img, restored, mask2d)
        region_stats = masked_region_stats(img, mask2d)

        if i < save_viz_n:
            fname = os.path.splitext(os.path.basename(row["path"]))[0]
            out_path = os.path.join(VERIFY_DIR, f"palette_{fname}_m0_{merge_mode}.png")
            visualize(img, damaged, restored, mask2d, out_path, vmax)
            print(f"    시각화 저장 → {out_path}")

        results.append({
            "path": row["path"], "mask_ratio": float(mask2d.mean()),
            "coverage": coverage, "fallback_px": fallback_px,
            "masked_l1": l1, "masked_rmse": rmse,
            "masked_psnr": psnr_val, "masked_ssim": ssim_val,
            "region_mean": region_stats["mean"],
        })

        elapsed_so_far = time.time() - t0
        eta_sec = (elapsed_so_far / (i + 1)) * (n_images - (i + 1))
        print(f"  [{i+1}/{n_images}] coverage={coverage:.3f} fallback_px={fallback_px} "
              f"masked_l1={l1:.2f} psnr={psnr_val:.2f}dB "
              f"(누적 {elapsed_so_far/3600:.2f}h, 예상잔여 {eta_sec/3600:.2f}h)")

        # --- 중간 저장: 장시간 실행 중 프로세스가 죽어도 결과 보존 ---
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n_images:
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f"    [중간 저장] {i+1}장까지 결과 저장됨 → {RESULTS_PATH}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n[완료] Palette(MISS1) 평가 결과 {len(res_df)}건 저장 → {RESULTS_PATH}")
    print(f"  merge_mode = {merge_mode}")
    print(f"  평균 coverage    = {res_df['coverage'].mean():.4f}")
    if merge_mode == "core_crop":
        print(f"  fallback 사용 픽셀 총합 = {total_fallback_pixels} (0에 가까울수록 이상적)")
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
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--merge_mode", type=str, default="average",
                     choices=["average", "best_tile", "core_crop"])
    args = ap.parse_args()
    main(ckpt_path=args.ckpt, n_images=args.n_images,
         save_viz_n=args.save_viz_n, seed=args.seed,
         sample_steps=args.sample_steps, merge_mode=args.merge_mode)