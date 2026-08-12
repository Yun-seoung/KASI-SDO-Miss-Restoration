"""
evaluate_ae.py — 학습된 AE로 MISS1 val 200장을 복원하고 baseline과 동일 지표로 평가.

[알고리즘 1] 학습 patch 생성 시 손상 덩어리 단위 crop 적용됨(precompute_patches.py)
[알고리즘 2] 겹치는 타일 병합 전략 선택 가능: --merge_mode average|best_tile|core_crop
[core_crop] 박사님 피드백 반영: 가중평균으로 인한 밴드(띠) 아티팩트 제거.
            패치 크기 L=512, stride=L/2(=256)로 추론하고, 각 패치의 중심부
            L/2(=256)만 남기고 가장자리 L/4(=128)씩 잘라낸 뒤 이어붙임.
            core 크기(L/2) == stride(L/2)이므로 core끼리 겹침/빈틈 없이
            정확히 타일링되어 평균을 낼 필요가 없어짐(=밴드 원천 차단).
            core로 커버되지 않는 극소수 경계 픽셀만 인접 타일 단순평균으로 보완.
[core grid 수정] core_crop 모드는 박스별 grid 대신 이미지 전체 기준 원점(0)
            고정 전역 grid(_core_grid_starts)를 사용 — 박스마다 grid가 어긋나
            core끼리 안 맞물리는 문제를 방지.
[seam correction 미적용] 덧셈형/affine형 둘 다 지표·시각 품질 개선 효과가 없어
            제외 — LaMa/VAE와 동일하게 순정 core_crop만 적용.
[버그 수정] 최종 합성 시 core/패치 사각형 전체가 아니라 실제 손상 픽셀
            (mask_full>0)에만 덮어쓰도록 수정 — 그렇지 않으면 멀쩡한 원본
            픽셀까지 모델 출력으로 덮어써서 블록 모양의 큰 아티팩트가 생김.
[fallback 지표 수정] fallback_pixel_count를 core_crop_merge 내부에서 세지 않고
            fallback_mask를 반환받아 실제 손상 픽셀(mask_full>0)과 교집합한
            개수만 세도록 수정 — 그렇지 않으면 손상과 무관한 패치 margin
            영역까지 다 세어져서 지표가 실제 위험도를 과대평가함.
[SSIM] masked_ssim 지표 포함

마스크는 MISS1 단계 마스크뱅크(mask_bank_miss1.npy) 사용.

실행:
  cd /NAS/ioGuard3/vol3/spaceai/cnu_ck_sdo_recov/sun_miss1_recov/ae
  /usr/bin/python -u evaluate_ae.py --ckpt runs/ckpt_ep50.pt --n_images 200 --save_viz_n 5 --merge_mode core_crop
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
from model import AE
from restore_cubic import masked_l1, masked_rmse, masked_psnr, masked_region_stats, masked_ssim

VAL_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_val_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss1.npy")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "ae_results.csv")
VERIFY_DIR = os.path.join(ROOT_DIR, "verify_out")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
os.makedirs(VERIFY_DIR, exist_ok=True)

PATCH_SIZE = 512
STRIDE = 128  # average / best_tile 모드용 (기존과 동일하게 유지, 비교 실험용)

# --- core_crop 모드 전용 파라미터 ---
CORE_MARGIN = PATCH_SIZE // 4   # 128, 패치 가장자리에서 잘라낼 폭 (L/4)
CORE_STRIDE = PATCH_SIZE // 2   # 256, core_crop 모드에서 사용할 stride (L/2)
# core 크기 = PATCH_SIZE - 2*CORE_MARGIN = 256 = CORE_STRIDE 이므로
# core끼리 겹침/빈틈 없이 정확히 이어붙여짐.


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
    """core_crop 전용: 이미지 전체 기준 원점(0) 고정, stride 간격의 단일 전역 grid.
    마지막 한 칸(이미지 경계)만 어긋나고 나머지는 항상 정확히 stride 간격 유지."""
    if total <= patch_size:
        return [0]
    starts = list(range(0, total - patch_size + 1, stride))
    if starts[-1] != total - patch_size:
        starts.append(total - patch_size)
    return starts


def core_crop_merge(h, w, tile_results, patch_size, core_margin):
    """
    가중평균 대신 각 패치의 중심부(core)만 사용해 이어붙임.
    fallback_mask: core로 못 채워 인접 타일 단순평균으로 보완한 픽셀 위치.
    """
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

    # --- fallback: core로 못 채운 픽셀 보완 ---
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

    return merged, covered, fill_mask  # fallback_pixel_count 대신 mask 자체를 반환


def restore_with_ae(model, img_full, mask_full, device, merge_mode="average",
                     patch_size=PATCH_SIZE, stride=None):
    """
    [알고리즘 1] 손상 덩어리 단위로 crop 위치 선정.
    [알고리즘 2] merge_mode에 따라 겹치는 타일을 가중평균 / 경성선택 / core_crop으로 병합.
    [core grid 수정] core_crop 모드는 박스와 무관하게 이미지 전체 기준 전역 grid 사용.
    """
    if stride is None:
        stride = CORE_STRIDE if merge_mode == "core_crop" else STRIDE

    h, w = img_full.shape
    boxes = get_damage_components(mask_full)
    if not boxes:
        return img_full.copy(), 1.0, 0

    seen = set()
    tile_results = []
    model.eval()

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
            damaged_n = img_n.copy()
            damaged_n[mask_patch > 0] = 0.0

            input_t = torch.from_numpy(damaged_n).unsqueeze(0).unsqueeze(0).float().to(device)
            pred_n = model(input_t).squeeze(0).squeeze(0).cpu().numpy()
            pred_denorm = denormalize(pred_n, ref_min, ref_max)

            tile_results.append((rs, cs, pred_denorm, mask_patch))

    if not tile_results:
        return img_full.copy(), 1.0, 0

    fallback_pixel_count = 0
    if merge_mode == "core_crop":
        merged, covered, fallback_mask = core_crop_merge(h, w, tile_results, patch_size, CORE_MARGIN)
        # 손상과 무관한 패치 margin 영역까지 세지 않도록, 실제 손상 픽셀(mask_full>0)과
        # 교집합한 개수만 카운트.
        fallback_pixel_count = int((fallback_mask & (mask_full > 0)).sum())
    else:
        merged, covered = merge_tiles(h, w, tile_results, merge_mode=merge_mode)

    # covered는 core/패치 사각형 전체를 가리키므로, 실제 손상 픽셀(mask_full>0)과
    # 교집합을 취해야 함. 그렇지 않으면 멀쩡한 원본 픽셀까지 모델 출력으로 덮어써서
    # 블록 모양의 큰 아티팩트가 생김.
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
                  "AE Restored (full, downsampled)", "Abs error (full, downsampled)"]
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

    titles_bot = ["Original (zoom)", "Damaged (zoom)", "AE Restored (zoom)", "Abs error (zoom)"]
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

    model = AE(base_ch=32).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
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

        restored, coverage, fallback_px = restore_with_ae(model, img, mask2d, device, merge_mode=merge_mode)
        total_fallback_pixels += fallback_px

        l1 = masked_l1(img, restored, mask2d)
        rmse = masked_rmse(img, restored, mask2d)
        psnr_val = masked_psnr(img, restored, mask2d)
        ssim_val = masked_ssim(img, restored, mask2d)
        region_stats = masked_region_stats(img, mask2d)

        if i < save_viz_n:
            fname = os.path.splitext(os.path.basename(row["path"]))[0]
            out_path = os.path.join(VERIFY_DIR, f"ae_{fname}_m0_{merge_mode}.png")
            visualize(img, damaged, restored, mask2d, out_path, vmax)
            print(f"    시각화 저장 → {out_path}")

        results.append({
            "path": row["path"], "mask_ratio": float(mask2d.mean()),
            "coverage": coverage, "fallback_px": fallback_px,
            "masked_l1": l1, "masked_rmse": rmse,
            "masked_psnr": psnr_val, "masked_ssim": ssim_val,
            "region_mean": region_stats["mean"],
        })

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{n_images} (직전 coverage={coverage:.3f}, fallback_px={fallback_px})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n[완료] AE(MISS1) 평가 결과 {len(res_df)}건 저장 → {RESULTS_PATH}")
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
    ap.add_argument("--merge_mode", type=str, default="average",
                     choices=["average", "best_tile", "core_crop"])
    args = ap.parse_args()
    main(ckpt_path=args.ckpt, n_images=args.n_images,
         save_viz_n=args.save_viz_n, seed=args.seed, merge_mode=args.merge_mode)