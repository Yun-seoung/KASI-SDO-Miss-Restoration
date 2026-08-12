"""
make_comparison_miss0.py — MISS0 baseline/AE/VAE/LaMa/Palette 비교를 하나의 이미지로 합성.
빨간 박스는 테두리만(채우지 않음), 손상 덩어리별로 정확한 크기로 표시.
왼쪽 = Damaged(전체) 위 / Original(zoom, 손상 전) 아래.
오른쪽 = 방법별 [복원 zoom | 오차 zoom] 5행 (Baseline/AE/VAE/LaMa/Palette).

[갱신] Palette 추가. MISS0의 restore_with_palette는 merge_mode 인자를 받지 않는
       구버전 인터페이스(가중평균 고정) — AE/VAE/LaMa와 달리 merge_mode를 넘기지
       않고 호출. MISS0는 손상 크기가 작아 core_crop 없이도 문제없다고 판단해
       이 구조를 그대로 유지.

"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(NANUM_PATH):
    fm.fontManager.addfont(NANUM_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT_DIR, "verify_out", "comparison_miss0_seed7.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

LAMA_REPO_PATH = "/path/to/lama"
if LAMA_REPO_PATH not in sys.path:
    sys.path.insert(0, LAMA_REPO_PATH)

PALETTE_REPO_PATH = "/path/to/Palette-Image-to-Image-Diffusion-Models"
if PALETTE_REPO_PATH not in sys.path:
    sys.path.insert(0, PALETTE_REPO_PATH)

SAMPLE_STEPS = 200  # Palette 전용: diffusion 샘플링 스텝 수


def load_module(name, path, extra_syspath=None):
    if extra_syspath:
        sys.path.insert(0, extra_syspath)
    for mod_name in ["model", "patch_geometry", "mask_loader", "restore_cubic",
                      "patch_dataset", "sun_lama", "sun_palette"]:
        sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if extra_syspath and extra_syspath in sys.path:
        sys.path.remove(extra_syspath)
    return mod


def get_crop_bounds(mask2d, margin=60):
    h, w = mask2d.shape
    damaged_rows = np.where(mask2d.any(axis=1))[0]
    r0 = max(0, damaged_rows.min() - margin)
    r1 = min(h, damaged_rows.max() + margin)
    damaged_cols = np.where(mask2d.any(axis=0))[0]
    c0 = max(0, damaged_cols.min() - 20)
    c1 = min(w, damaged_cols.max() + 20)
    return r0, r1, c0, c1


def minmax_diff(orig, restored):
    o_min, o_max = float(orig.min()), float(orig.max())
    denom = max(o_max - o_min, 1e-8)
    orig_n = (orig.astype(np.float32) - o_min) / denom
    restored_n = (restored.astype(np.float32) - o_min) / denom
    return np.abs(orig_n - restored_n)


def draw_component_boxes(ax, boxes, offset_r=0, offset_c=0, color="red", lw=1.5):
    """손상 덩어리별로 정확한 크기의 테두리만 그리기 (채우지 않음, fill=False)."""
    for (br0, br1, bc0, bc1) in boxes:
        local_r0 = br0 - offset_r
        local_c0 = bc0 - offset_c
        ax.add_patch(Rectangle(
            (local_c0, local_r0), bc1 - bc0, br1 - br0,
            fill=False, edgecolor=color, linewidth=lw))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, os.path.join(ROOT_DIR, "baseline"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
    from mask_loader import MaskBank
    from patch_geometry import get_damage_components
    from restore_cubic import restore_vertical_cubic, apply_damage, masked_l1, masked_psnr

    ae_eval = load_module("ae_eval_miss0", os.path.join(ROOT_DIR, "ae", "evaluate_ae.py"),
                           extra_syspath=os.path.join(ROOT_DIR, "ae"))
    vae_eval = load_module("vae_eval_miss0", os.path.join(ROOT_DIR, "vae", "evaluate_vae.py"),
                            extra_syspath=os.path.join(ROOT_DIR, "vae"))
    lama_eval = load_module("lama_eval_miss0", os.path.join(ROOT_DIR, "lama", "evaluate_lama.py"),
                             extra_syspath=os.path.join(ROOT_DIR, "lama"))
    palette_eval = load_module("palette_eval_miss0", os.path.join(ROOT_DIR, "palette", "evaluate_palette.py"),
                                extra_syspath=os.path.join(ROOT_DIR, "palette"))

    VAL_CSV = os.path.join(ROOT_DIR, "data", "normal_paths_val_cached.csv")
    MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", "mask_bank_miss0.npy")
    AE_CKPT = os.path.join(ROOT_DIR, "ae", "runs", "ckpt_ep50.pt")
    VAE_CKPT = os.path.join(ROOT_DIR, "vae", "runs", "ckpt_ep50.pt")
    LAMA_CKPT = os.path.join(ROOT_DIR, "lama", "runs", "ckpt_ep160.pt")  # 실제 최신 파일명 확인 후 조정
    PALETTE_CKPT = os.path.join(ROOT_DIR, "palette", "runs", "ckpt_ep50.pt")  # 실제 최신 파일명 확인 후 조정

    seed = 7
    val_df = pd.read_csv(VAL_CSV).sample(n=200, random_state=seed).reset_index(drop=True)
    row = val_df.iloc[0]
    img = np.load(row["cache_path"])
    vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0

    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=seed)
    mask2d = mask_bank.sample()
    damaged = apply_damage(img, mask2d, fill_value=0.0)

    restored_baseline = restore_vertical_cubic(damaged, mask2d)

    model_ae = ae_eval.AE(base_ch=32).to(device)
    ckpt = torch.load(AE_CKPT, map_location=device)
    model_ae.load_state_dict(ckpt["model"])
    restored_ae, _ = ae_eval.restore_with_ae(model_ae, img, mask2d, device, merge_mode="average")

    model_vae = vae_eval.VAE(base_ch=32).to(device)
    ckpt = torch.load(VAE_CKPT, map_location=device)
    model_vae.load_state_dict(ckpt["model"])
    restored_vae, _ = vae_eval.restore_with_vae(model_vae, img, mask2d, device, merge_mode="average")

    G = lama_eval.build_generator(img_ch=1).to(device)
    ckpt = torch.load(LAMA_CKPT, map_location=device)
    G.load_state_dict(ckpt["G"])
    restored_lama, _ = lama_eval.restore_with_lama(
        G, img, mask2d, device, index=0, seed=seed, merge_mode="average"
    )

    net_palette = palette_eval.build_palette_network(device=device)
    ckpt = torch.load(PALETTE_CKPT, map_location=device)
    net_palette.load_state_dict(ckpt["net"])
    palette_eval.set_eval_noise_schedule(net_palette, device=device, n_timestep=SAMPLE_STEPS)
    # MISS0의 restore_with_palette는 merge_mode 인자를 받지 않음 (구버전, 가중평균 고정)
    restored_palette, _ = palette_eval.restore_with_palette(
        net_palette, img, mask2d, device
    )

    r0, r1, c0, c1 = get_crop_bounds(mask2d)
    boxes = get_damage_components(mask2d)

    methods = [
        ("Baseline", restored_baseline),
        ("AE", restored_ae),
        ("VAE", restored_vae),
        ("LaMa", restored_lama),
        ("Palette", restored_palette),
    ]
    n_rows = len(methods)

    def downsample(a, factor=8):
        return a[::factor, ::factor]

    ds_factor = 8

    fig = plt.figure(figsize=(16, 2.0 * n_rows))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.1)

    # ── 왼쪽: Damaged(전체) 위, Original(zoom) 아래 — 오른쪽 zoom 1행 크기와 맞춤 ──
    left = outer[0].subgridspec(2, 1, hspace=0.15, height_ratios=[n_rows - 1, 1])

    ax_damaged = fig.add_subplot(left[0])
    ax_damaged.imshow(downsample(img), cmap="gray", vmin=0, vmax=vmax)
    ax_damaged.set_title("Damaged (전체, 확대 영역 표시)", fontsize=11)
    draw_component_boxes(ax_damaged, [(b[0] / ds_factor, b[1] / ds_factor,
                                        b[2] / ds_factor, b[3] / ds_factor) for b in boxes],
                          color="red", lw=2)
    ax_damaged.axis("off")

    ax_orig_zoom = fig.add_subplot(left[1])
    ax_orig_zoom.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=vmax,
                         interpolation="nearest", aspect="auto")
    draw_component_boxes(ax_orig_zoom, boxes, offset_r=r0, offset_c=c0, color="red", lw=1.5)
    ax_orig_zoom.set_title("Original (zoom, 손상 전)", fontsize=10)
    ax_orig_zoom.axis("off")

    # ── 오른쪽: 방법별 [복원 zoom | 오차 zoom], 5행 ──
    inner = outer[1].subgridspec(n_rows, 2, hspace=0.35, wspace=0.05)

    for i, (name, restored) in enumerate(methods):
        diff = minmax_diff(img, restored)
        zoom_restored = restored[r0:r1, c0:c1]
        zoom_diff = diff[r0:r1, c0:c1]
        zoom_mask_bool = mask2d[r0:r1, c0:c1] > 0
        diff_vmax = max(np.percentile(zoom_diff[zoom_mask_bool], 95), 1e-6) if zoom_mask_bool.any() else 1e-3

        l1 = masked_l1(img, restored, mask2d)
        psnr_val = masked_psnr(img, restored, mask2d)

        ax_r = fig.add_subplot(inner[i, 0])
        ax_r.imshow(zoom_restored, cmap="gray", vmin=0, vmax=vmax,
                    interpolation="nearest", aspect="auto")
        draw_component_boxes(ax_r, boxes, offset_r=r0, offset_c=c0, color="red", lw=1.5)
        ax_r.set_title(f"{name} Restored (zoom)", fontsize=10)
        ax_r.axis("off")

        ax_e = fig.add_subplot(inner[i, 1])
        ax_e.imshow(zoom_diff, cmap="inferno", vmin=0, vmax=diff_vmax,
                    interpolation="nearest", aspect="auto")
        ax_e.set_title(f"{name} Abs error | l1={l1:.2f} PSNR={psnr_val:.2f}dB", fontsize=10)
        ax_e.axis("off")

    plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[완료] 저장 → {OUT_PATH}")


if __name__ == "__main__":
    main()