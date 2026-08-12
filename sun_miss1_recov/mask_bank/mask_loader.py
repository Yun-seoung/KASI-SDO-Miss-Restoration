"""
mask_loader.py — 실측 마스크뱅크(mask_bank_missX.npy) 로더.

sample(): 평가용. 4096x4096 전체 실측 마스크 하나를 무작위로 반환.

sample_local(patch_size, ratio_pool): 학습용. patch 안에서 직접 무작위
  밴드형 마스크 생성. 손상비율을 고정값이 아니라 ratio_pool(offline
  patch_cache에서 실측한 손상비율 분포, 0.4%~100%까지 폭넓게 분포)에서
  매번 무작위로 뽑아 사용 -> 극단적으로 좁은 patch와 완전히 뒤덮인 patch를
  모두 학습하게 되어 실전 다양성을 재현.

  [문제 이력]
  1) 손상 위치를 patch 중앙에 고정 배치 -> 과포화(47.6%) 문제로 폐기.
  2) 손상 행을 흩어지게 생성 -> 실제 밴드보다 너무 쉬워서 성능 저하.
  3) 연속 밴드 + 고정 목표비율(10.4%) -> offline 대비 여전히 부족(l1 8.13
     vs 6.59). 원인: offline은 손상비율이 0.39%~100%까지 폭넓게 분포하는데
     (같은 넓은 손상 덩어리를 슬라이딩하며 만들다 보니 가장자리~한가운데까지
     전 범위가 자연히 나옴), online은 항상 10% 근처로만 고정해서 극단적
     케이스(거의 무손상 ~ 완전 뒤덮임)를 전혀 학습 못 했음.
  4) [현재] ratio_pool로 offline의 실측 분포를 그대로 재현.
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

    def sample_local(self, patch_size, rng=None, ratio_pool=None, target_ratio=None):
        """
        학습용: patch 안에서 직접 무작위 밴드형 마스크 생성.

        ratio_pool: offline 실측 손상비율 분포(array). 지정하면 매번 이 중
                    하나를 무작위로 뽑아 사용 (권장, 극단치까지 재현).
        target_ratio: ratio_pool이 없을 때 쓸 고정 비율. 둘 다 없으면
                      self.avg_ratio(이미지 전체 평균) 사용.
        """
        rng = rng if rng is not None else self.rng
        mask = np.zeros((patch_size, patch_size), dtype=np.float32)

        if ratio_pool is not None and len(ratio_pool) > 0:
            ratio = float(ratio_pool[rng.randint(len(ratio_pool))])
        elif target_ratio is not None:
            ratio = target_ratio
        else:
            ratio = self.avg_ratio

        target_pixels = int(patch_size * patch_size * ratio)
        if target_pixels == 0:
            return mask

        n_rows_needed = max(1, target_pixels // patch_size)
        n_rows_needed = min(n_rows_needed, patch_size)

        start_row = rng.randint(0, patch_size - n_rows_needed + 1)
        mask[start_row:start_row + n_rows_needed, :] = 1.0
        return mask