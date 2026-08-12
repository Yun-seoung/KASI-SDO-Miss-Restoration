"""
train_palette.py — 원본 Palette Network(guided_diffusion U-Net)를
정상 태양 이미지 + MISS1 단계 마스크뱅크로 온라인 patch 샘플링(ratio_pool 기반)
방식으로 학습. AE/VAE/LaMa에서 검증된 최종 파이프라인과 동일.

[수정 2026-07] patch_cache(고정 3500개, ratio_pool 개선 전 생성분) 대신
  OnlinePatchDataset으로 교체. LaMa와 동일하게 매 스텝 위치·마스크를
  실시간으로 새로 샘플링하도록 통일. patch_cache 의존성 완전 제거.

주의:
  - Docker 컨테이너 /dev/shm 제약으로 num_workers=0 고정 필요
  - PYTHONPATH에 Palette 저장소 루트 필요
  - MISS1은 손상 비율(2.66%)이 MISS0(0.28%)의 약 9.5배라 GPU 메모리 여유가
    더 빠듯할 수 있음. batch=4에서 시작, OOM 나면 batch=1까지 낮출 것.
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
ROOT_DIR = os.path.dirname(THIS_DIR)  # sun_miss1_recov/
SIBLING_ROOT = os.path.dirname(ROOT_DIR)  # cnu_ck_sdo_recov/
MISS0_DATA_DIR = os.path.join(SIBLING_ROOT, "sun_miss0_recov", "data")
AE_DIR = os.path.join(ROOT_DIR, "ae")  # online_patch_dataset.py 위치

sys.path.insert(0, os.path.join(ROOT_DIR, "mask_bank"))
sys.path.insert(0, AE_DIR)
sys.path.insert(0, THIS_DIR)

from mask_loader import MaskBank  # noqa: E402
from online_patch_dataset import OnlinePatchDataset  # noqa: E402
from sun_palette import build_palette_network  # noqa: E402

TRAIN_CSV = os.path.join(MISS0_DATA_DIR, "normal_paths_train_cached.csv")
# LaMa와 동일한 방식으로 mask_bank_miss* 파일을 자동 탐색 (하드코딩 방지)
MASK_BANK_PATH = os.path.join(ROOT_DIR, "mask_bank", os.path.basename(
    [f for f in os.listdir(os.path.join(ROOT_DIR, "mask_bank")) if f.startswith("mask_bank_miss")][0]
))
CKPT_DIR = os.path.join(THIS_DIR, "runs")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.csv")
os.makedirs(CKPT_DIR, exist_ok=True)

NOISE_STD = 1.0  # 프로젝트 전체(LaMa) 확정값과 통일


def make_cond_image(gt_image, mask, noise_std=NOISE_STD):
    """마스킹 영역을 노이즈로 채운 조건 이미지 (LaMa의 make_lama_input과 동일 철학)."""
    noise = torch.randn_like(gt_image) * noise_std
    return torch.where(mask.bool(), noise, gt_image)


def main(epochs=50, batch=4, lr=1e-4, ckpt_every=10, resume_from=None,
         num_workers=0, patches_per_image=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, mask_bank={os.path.basename(MASK_BANK_PATH)}")

    train_paths = pd.read_csv(TRAIN_CSV)["cache_path"].tolist()
    mask_bank = MaskBank(path=MASK_BANK_PATH, seed=0)

    train_ds = OnlinePatchDataset(
        image_cache_paths=train_paths, mask_bank=mask_bank, patch_size=512,
        patches_per_image=patches_per_image, seed=0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                               num_workers=num_workers, drop_last=True)

    net = build_palette_network(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] epoch {start_epoch}부터 재개: {resume_from}")

    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "loss", "seconds"])

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)  # [추가] LaMa와 동일하게 매 epoch 재셔플
        net.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        # OnlinePatchDataset 반환: (input_t, target_t, mask_t)
        # target_t = 정상 정규화 이미지(y_0), mask_t = {0,1} 마스크 (실시간 샘플링)
        for _input_t, target_t, mask_t in train_loader:
            y_0 = target_t.to(device)
            mask = mask_t.to(device)
            y_cond = make_cond_image(y_0, mask)

            loss = net(y_0=y_0, y_cond=y_cond, mask=mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        print(f"[epoch {epoch+1}/{epochs}] loss={avg_loss:.4f} elapsed={elapsed:.1f}s")
        log_writer.writerow([epoch + 1, avg_loss, elapsed])
        log_file.flush()

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"ckpt_ep{epoch+1}.pt")
            torch.save({
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, ckpt_path)
            print(f"    체크포인트 저장 → {ckpt_path}")

    log_file.close()
    print("\n[완료] Palette(MISS1) 학습 종료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--patches_per_image", type=int, default=10)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         ckpt_every=args.ckpt_every, resume_from=args.resume_from,
         num_workers=args.num_workers, patches_per_image=args.patches_per_image)