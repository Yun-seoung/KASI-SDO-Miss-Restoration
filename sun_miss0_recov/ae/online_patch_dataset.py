"""
online_patch_dataset.py — 학습용 온라인 랜덤 patch 샘플링 Dataset.

[설계] 매 __getitem__ 호출마다 즉석으로 (이미지, 위치)를 무작위로 뽑고,
그 patch 안에 mask_bank.sample_local()로 직접 무작위 마스크를 생성한다.
patch 위치와 마스크 생성이 완전히 독립적이라, 손상 위치 편향이 없다.

patches_per_image: 매 epoch마다 각 이미지가 정확히 이 횟수만큼 등장하도록
  index_pool을 구성 후 셔플 (노출 횟수 균등 보장, 위치·마스크는 매번 무작위).

이미지는 __init__에서 한 번에 RAM 프리로드
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def normalize_01(img2d, eps=1e-8):
    ref_min = float(img2d.min())
    ref_max = float(img2d.max())
    denom = max(ref_max - ref_min, eps)
    return (img2d.astype(np.float32) - ref_min) / denom


class OnlinePatchDataset(Dataset):
    def __init__(self, image_cache_paths, mask_bank, patch_size=256,
                 patches_per_image=10, seed=0):
        self.mask_bank = mask_bank
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.rng = np.random.RandomState(seed)

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
        r0 = self.rng.randint(0, self.H - ps + 1)
        c0 = self.rng.randint(0, self.W - ps + 1)
        img_patch = img_full[r0:r0 + ps, c0:c0 + ps]

        mask_patch = self.mask_bank.sample_local(ps, rng=self.rng)

        img_n = normalize_01(img_patch)
        mask_t = mask_patch.astype(np.float32)

        damaged_n = img_n.copy()
        damaged_n[mask_t > 0] = 0.0

        input_t = torch.from_numpy(damaged_n).unsqueeze(0).float()
        target_t = torch.from_numpy(img_n).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_t).unsqueeze(0).float()

        return input_t, target_t, mask_tensor