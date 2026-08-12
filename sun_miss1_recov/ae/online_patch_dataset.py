"""
online_patch_dataset.py — 학습용 온라인 랜덤 patch 샘플링 Dataset (MISS1).

손상비율은 eval_aligned_ratio_pool.npy(평가 슬라이딩 윈도우 실측 분포)에서 매번
무작위로 뽑아 사용. 기존 offline_ratio_pool.npy는 patch 전체 완전손상(ratio≈1.0)
극단치 비중이 높고 중간~고손상(20~75%) 구간 다양성이 부족해, 실제 평가 시점
슬라이딩 윈도우가 만들어내는 분포(generate_eval_aligned_ratio_pool.py로 생성)로
교체함. 배경(태양 밖) patch는 재시도로 회피.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

BACKGROUND_MEAN_THRESHOLD = 5.0
MAX_BACKGROUND_RETRIES = 5

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RATIO_POOL_PATH = os.path.join(THIS_DIR, "eval_aligned_ratio_pool.npy")


def normalize_01(img2d, eps=1e-8):
    ref_min = float(img2d.min())
    ref_max = float(img2d.max())
    denom = max(ref_max - ref_min, eps)
    return (img2d.astype(np.float32) - ref_min) / denom


class OnlinePatchDataset(Dataset):
    def __init__(self, image_cache_paths, mask_bank, patch_size=512,
                 patches_per_image=10, seed=0, ratio_pool_path=RATIO_POOL_PATH):
        self.mask_bank = mask_bank
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.rng = np.random.RandomState(seed)

        if os.path.exists(ratio_pool_path):
            self.ratio_pool = np.load(ratio_pool_path)
            print(f"[OnlinePatchDataset] ratio_pool 로드: {ratio_pool_path} "
                  f"({len(self.ratio_pool)}개, 평균={self.ratio_pool.mean()*100:.2f}%, "
                  f"중앙값={np.median(self.ratio_pool)*100:.2f}%)")
        else:
            self.ratio_pool = None
            print(f"[OnlinePatchDataset] [경고] ratio_pool 없음({ratio_pool_path}) "
                  f"-> mask_bank.avg_ratio 고정값 사용. "
                  f"generate_eval_aligned_ratio_pool.py를 먼저 실행하세요.")

        print(f"[OnlinePatchDataset] 이미지 {len(image_cache_paths)}장 RAM 프리로드 시작...")
        self.images = []
        for p in tqdm(image_cache_paths, desc="프리로드"):
            self.images.append(np.load(p).astype(np.float32))
        self.H, self.W = self.images[0].shape
        total_mb = sum(im.nbytes for im in self.images) / (1024 ** 2)
        print(f"[OnlinePatchDataset] 프리로드 완료: {len(self.images)}장, "
              f"총 {total_mb:.1f}MB in RAM, 크기 {self.H}x{self.W}")

        self._build_index_pool()
        total = len(self.images) * patches_per_image
        print(f"[OnlinePatchDataset] 이미지당 정확히 {patches_per_image}개 (총 {total}개/epoch)")

    def _build_index_pool(self):
        self.index_pool = np.repeat(np.arange(len(self.images)), self.patches_per_image)
        self.rng.shuffle(self.index_pool)

    def set_epoch(self, epoch):
        """매 epoch 시작 시 호출 -> 이미지 등장 순서 재셔플 (노출 횟수는 항상 균등)."""
        self._build_index_pool()

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx):
        ps = self.patch_size
        img_idx = self.index_pool[idx]
        img_full = self.images[img_idx]

        for _ in range(MAX_BACKGROUND_RETRIES):
            r0 = self.rng.randint(0, self.H - ps + 1)
            c0 = self.rng.randint(0, self.W - ps + 1)
            img_patch = img_full[r0:r0 + ps, c0:c0 + ps]
            if img_patch.mean() >= BACKGROUND_MEAN_THRESHOLD:
                break

        mask_patch = self.mask_bank.sample_local(ps, rng=self.rng, ratio_pool=self.ratio_pool)

        img_n = normalize_01(img_patch)
        mask_t = mask_patch.astype(np.float32)

        damaged_n = img_n.copy()
        damaged_n[mask_t > 0] = 0.0

        input_t = torch.from_numpy(damaged_n).unsqueeze(0).float()
        target_t = torch.from_numpy(img_n).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_t).unsqueeze(0).float()

        return input_t, target_t, mask_tensor