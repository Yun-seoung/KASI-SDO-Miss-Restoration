"""
mask_loader.py — 실측 마스크뱅크(mask_bank_missX.npy) 로더.

sample(): 기존 방식 그대로, 4096x4096 전체 마스크 하나를 무작위로 반환.
          (evaluate_*.py에서 여전히 사용)

sample_local(patch_size): [재설계] patch 안에서 직접 무작위 마스크를 생성.
  마스크뱅크의 특정 좌표를 참조하지 않고, patch 위치와 무관하게 완전히
  새로 그림. 실제 MISS 손상 특성(가로 행 단위로 통째로 날아감)은 유지하되,
  어느 행이 손상될지는 매번 무작위로 정하고, 손상 픽셀 비율이 이 마스크뱅크의
  실측 평균 비율(self.avg_ratio, MISS0=0.28%/MISS1=2.66%)에 가깝도록
  손상 행 개수를 조절한다.

  [이전 버그 이력] 손상 위치를 patch 중앙에 고정 배치하던 방식은 MISS1처럼
  손상 밴드가 넓은 경우 patch 손상비율이 평균 47.6%까지 치솟아 학습 실패
  (psnr 24dB대 정체)를 유발했음. 이번 재설계로 patch 위치와 마스크 생성을
  완전히 분리해서 근본적으로 해결.
"""

import numpy as np

class MaskBank:
    def __init__(self, path, seed=0):
        self.masks = np.load(path)  # (N, 4096, 4096) bool 또는 {0,1}
        self.rng = np.random.RandomState(seed)
        self.avg_ratio = float(self.masks.mean())
        print(f"[MaskBank] 로드 완료: {self.masks.shape}, 평균 손상비율={self.avg_ratio:.4f}")

    def sample(self):
        """평가용: 4096x4096 전체 실측 마스크 하나를 무작위로 반환 (변경 없음)."""
        idx = self.rng.randint(len(self.masks))
        return self.masks[idx].copy()

    def sample_local(self, patch_size, rng=None):
        """
        학습용: patch 안에서 직접 무작위 마스크 생성.
        목표 손상비율(self.avg_ratio)에 맞춰 손상 행 개수를 계산하고,
        그 행들을 patch 폭 전체에 걸쳐 손상시킨다 (가로줄 손상 특성 유지).
        """
        rng = rng if rng is not None else self.rng
        mask = np.zeros((patch_size, patch_size), dtype=np.float32)

        target_pixels = int(patch_size * patch_size * self.avg_ratio)
        if target_pixels == 0:
            return mask  # 목표 비율이 너무 낮으면 이번 patch는 무손상 (자연 발생)

        n_rows_needed = max(1, target_pixels // patch_size)
        n_rows_needed = min(n_rows_needed, patch_size)

        damaged_rows = rng.choice(patch_size, size=n_rows_needed, replace=False)
        mask[damaged_rows, :] = 1.0
        return mask