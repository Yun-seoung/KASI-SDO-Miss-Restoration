"""
train_palette.py — 원본 Palette Network(guided_diffusion U-Net)를
정상 태양 이미지 + MISS0 마스크뱅크 패치로 학습.
patch_cache는 ae/ 것을 그대로 재사용 (AE/VAE/LaMa와 동일 컨벤션).

원본 network.py의 forward(y_0, y_cond, mask, noise)가 내부에서
loss = loss_fn(mask*noise, mask*noise_hat)를 계산해서 반환하므로,
여기서는 그 forward를 호출하고 반환된 loss로 backward만 하면 됨.

주의:
  - Docker 컨테이너 /dev/shm 제약으로 num_workers=0 고정 필요
  - PYTHONPATH에 Palette 저장소 루트 필요

"""

import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
AE_DIR = os.path.join(ROOT_DIR, "ae")
sys.path.insert(0, AE_DIR)
sys.path.insert(0, THIS_DIR)

from patch_dataset import PrecomputedPatchDataset  # noqa: E402  (ae/patch_dataset.py)
from sun_palette import build_palette_network         # noqa: E402

PATCH_DIR = os.path.join(AE_DIR, "patch_cache")
CKPT_DIR = os.path.join(THIS_DIR, "runs")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.csv")
os.makedirs(CKPT_DIR, exist_ok=True)

NOISE_STD = 1.0  # 프로젝트 전체(LaMa) 확정값과 통일


def make_cond_image(gt_image, mask, noise_std=NOISE_STD):
    """마스킹 영역을 노이즈로 채운 조건 이미지 (LaMa의 make_lama_input과 동일 철학)."""
    noise = torch.randn_like(gt_image) * noise_std
    return torch.where(mask.bool(), noise, gt_image)


def main(epochs=50, batch=8, lr=1e-4, ckpt_every=5, resume_from=None, num_workers=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_ds = PrecomputedPatchDataset(PATCH_DIR, tag="train")
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
        net.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        # PrecomputedPatchDataset 반환: (input_t, target_t, mask_t)
        # target_t = 정상 정규화 이미지(y_0), mask_t = {0,1} 마스크
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
    print("\n[완료] Palette 학습 종료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    main(epochs=args.epochs, batch=args.batch, lr=args.lr,
         ckpt_every=args.ckpt_every, resume_from=args.resume_from,
         num_workers=args.num_workers)