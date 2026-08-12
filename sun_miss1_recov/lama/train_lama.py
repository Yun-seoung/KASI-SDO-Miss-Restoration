"""
train_lama.py — LaMa를 정상 태양 이미지 + MISS1 마스크뱅크로 온라인 patch
샘플링(ratio_pool 기반) 방식으로 학습. AE/VAE에서 검증된 최종 파이프라인과 동일.

[재학습 개선 — 원인 분석 결과 반영]
기존 220epoch 학습 로그 분석 결과 adv/fm loss가 학습 내내 정체되어 있었음.
sun_lama.py에서 D의 norm_layer를 InstanceNorm2d로 교체(원인 A)했고, 여기서는
아래 두 가지를 추가로 적용:
  - lr을 1e-3 -> 1e-4로 하향 (adversarial 학습엔 1e-3이 과도하게 컸을 가능성)
  - warm-up: 처음 warmup_epochs 동안은 adv/fm=0(순수 l1만 학습)으로 G가 구조를
    먼저 잡게 한 뒤, ramp_epochs에 걸쳐 목표 가중치까지 선형 증가

주의: offline_ratio_pool.npy가 sun_miss1_recov/ae/ 폴더에 이미 있어야 함
(AE 재학습 때 생성해둔 것 재사용).
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
SIBLING_ROOT = os.path.dirname(ROOT_DIR)
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")
AE_DIR = os.path.join(ROOT_DIR, "ae")

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, AE_DIR)
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank  # noqa: E402
from online_patch_dataset import OnlinePatchDataset  # noqa: E402
from sun_lama import (  # noqa: E402
    build_generator, build_discriminator, LamaLosses, make_lama_input,
)

TRAIN_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_train_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", os.path.basename(
    [f for f in os.listdir(os.path.join(ROOT_DIR, "mask_bank")) if f.startswith("mask_bank_miss")][0]
))
# [변경] 기존 runs/(1e-3, BatchNorm, no warm-up 결과물)와 분리해서 새 실험을 별도 보관
CKPT_DIR = os.path.join(THIS_DIR, "runs_v3")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.csv")
os.makedirs(CKPT_DIR, exist_ok=True)

IMG_CH = 1
TRAIN_NOISE_STD = 1.0


def get_progress(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """epoch가 warmup_epochs 이하면 0.0(순수 l1),
    warmup_epochs~warmup_epochs+ramp_epochs 구간엔 0->1 선형 증가,
    그 이후는 1.0(목표 가중치 전부 적용)."""
    if epoch <= warmup_epochs:
        return 0.0
    if epoch >= warmup_epochs + ramp_epochs:
        return 1.0
    return (epoch - warmup_epochs) / max(1, ramp_epochs)


def main(epochs=220, batch=4, lr=1e-4, ckpt_every=20, resume_from=None,
         num_workers=0, patches_per_image=10, warmup_epochs=20, ramp_epochs=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, img_ch={IMG_CH}, noise_std={TRAIN_NOISE_STD}, "
          f"lr={lr}, warmup_epochs={warmup_epochs}, ramp_epochs={ramp_epochs}")

    train_paths = pd.read_csv(TRAIN_CSV)["cache_path"].tolist()
    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=0)

    train_ds = OnlinePatchDataset(
        image_cache_paths=train_paths, mask_bank=mask_bank, patch_size=512,
        patches_per_image=patches_per_image, seed=0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                               num_workers=num_workers, drop_last=True)

    G = build_generator(IMG_CH).to(device)
    D = build_discriminator(IMG_CH).to(device)
    losses = LamaLosses()  # w_adv_max=10.0, w_fm_max=100.0은 내부 기본값 유지
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt["epoch"]
        print(f"[resume] epoch {start_epoch}부터 재개: {resume_from}")

    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="")
    csv_fields = ["epoch", "l1", "adv", "fm", "d", "progress", "sec_per_epoch"]
    log_writer = csv.DictWriter(log_file, fieldnames=csv_fields)
    if not log_exists:
        log_writer.writeheader()
        log_file.flush()

    for ep in range(start_epoch + 1, epochs + 1):
        progress = get_progress(ep, warmup_epochs, ramp_epochs)
        losses.set_progress(progress)  # 이 epoch에서 쓸 adv/fm 가중치 갱신

        train_ds.set_epoch(ep)
        t0 = time.time()
        G.train(); D.train()
        agg = {"l1": 0.0, "adv": 0.0, "fm": 0.0, "d": 0.0}

        for _input_t, target_t, mask_t in tqdm(train_loader, total=len(train_loader)):
            img = target_t.to(device)
            mask = mask_t.to(device)

            pred = G(make_lama_input(img, mask, noise_std=TRAIN_NOISE_STD))
            opt_d.zero_grad(set_to_none=True)
            d_l = losses.d_loss(D, img, pred, mask)
            d_l.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            pred = G(make_lama_input(img, mask, noise_std=TRAIN_NOISE_STD))
            g_l, parts = losses.g_loss(D, pred, img, mask)
            g_l.backward()
            opt_g.step()

            for k in ("l1", "adv", "fm"):
                agg[k] += parts[k]
            agg["d"] += d_l.item()

        nb = len(train_loader)
        elapsed = time.time() - t0
        row = {
            "epoch": ep,
            "l1": round(agg["l1"] / nb, 5),
            "adv": round(agg["adv"] / nb, 5),
            "fm": round(agg["fm"] / nb, 5),
            "d": round(agg["d"] / nb, 5),
            "progress": round(progress, 3),
            "sec_per_epoch": round(elapsed, 1),
        }
        print(f"[ep {ep:03d}/{epochs}] progress={progress:.2f} " +
              " ".join(f"{k}={agg[k]/nb:.4f}" for k in agg) +
              f" ({elapsed:.1f}s)")

        log_writer.writerow(row)
        log_file.flush()

        if ep % ckpt_every == 0 or ep == epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"ckpt_ep{ep}.pt")
            torch.save({
                "epoch": ep, "G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            }, ckpt_path)
            print(f"    체크포인트 저장 → {ckpt_path}")

    log_file.close()
    print("\n[완료] LaMa 학습 종료. 평가는 evaluate_lama.py로 별도 실행하세요.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)  # [변경] 1e-3 -> 1e-4
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--patches_per_image", type=int, default=10)
    ap.add_argument("--warmup_epochs", type=int, default=20)  # [추가]
    ap.add_argument("--ramp_epochs", type=int, default=30)    # [추가]
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         ckpt_every=args.ckpt_every, resume_from=args.resume_from,
         num_workers=args.num_workers, patches_per_image=args.patches_per_image,
         warmup_epochs=args.warmup_epochs, ramp_epochs=args.ramp_epochs)