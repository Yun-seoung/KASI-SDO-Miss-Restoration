"""
train_ae.py — AE를 정상 태양 이미지 + 마스크뱅크로 온라인 patch 샘플링 방식으로 학습.

[변경] patch_cache(offline 고정 3000개) 대신 OnlinePatchDataset으로 교체.
매 epoch마다 다른 위치/마스크 조합의 patch를 즉석에서 샘플링.
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # sun_miss0_recov/ 또는 sun_miss1_recov/
SIBLING_ROOT = os.path.dirname(ROOT_DIR)
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank  # noqa: E402
from online_patch_dataset import OnlinePatchDataset  # noqa: E402
from model import AE, masked_l1_loss  # noqa: E402  (기존 model.py 그대로)
from utils_metrics import psnr  # noqa: E402  (기존 utils_metrics.py 그대로)

TRAIN_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_train_cached.csv")
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", os.path.basename(
    [f for f in os.listdir(os.path.join(ROOT_DIR, "mask_bank")) if f.startswith("mask_bank_miss")][0]
))
CKPT_DIR = os.path.join(THIS_DIR, "runs")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.csv")
os.makedirs(CKPT_DIR, exist_ok=True)


def main(epochs=50, batch=16, lr=1e-4, ckpt_every=5, resume_from=None,
         num_workers=0, patches_per_image=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_paths = pd.read_csv(TRAIN_CSV)["cache_path"].tolist()
    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=0)

    train_ds = OnlinePatchDataset(
        image_cache_paths=train_paths,
        mask_bank=mask_bank,
        patch_size=256,
        patches_per_image=patches_per_image,
        keep_empty_prob=0.05,
        seed=0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                               num_workers=num_workers, drop_last=True)

    model = AE(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] epoch {start_epoch}부터 재개: {resume_from}")

    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "l1_loss", "psnr", "seconds"])

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)   # 이 줄 추가 — 매 epoch마다 이미지 노출 순서 재셔플 (노출 횟수는 항상 균등)
        model.train()
        t0 = time.time()
        total_loss, total_psnr, n_batches = 0.0, 0.0, 0

        for input_t, target_t, mask_t in train_loader:
            input_t = input_t.to(device)
            target_t = target_t.to(device)
            mask_t = mask_t.to(device)

            pred = model(input_t)
            loss = masked_l1_loss(pred, target_t, mask_t, mask_weight=5.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                p = psnr(pred, target_t)

            total_loss += loss.item()
            total_psnr += p.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_psnr = total_psnr / max(n_batches, 1)
        elapsed = time.time() - t0

        print(f"[epoch {epoch+1}/{epochs}] l1_loss={avg_loss:.4f} "
              f"psnr={avg_psnr:.2f}dB elapsed={elapsed:.1f}s")
        log_writer.writerow([epoch + 1, avg_loss, avg_psnr, elapsed])
        log_file.flush()

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"ckpt_ep{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, ckpt_path)
            print(f"    체크포인트 저장 → {ckpt_path}")

    log_file.close()
    print("\n[완료] AE 학습 종료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--patches_per_image", type=int, default=10)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         ckpt_every=args.ckpt_every, resume_from=args.resume_from,
         num_workers=args.num_workers, patches_per_image=args.patches_per_image)