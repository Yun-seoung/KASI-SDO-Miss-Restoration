"""
patch_geometry.py — 패치 crop 위치 선정(알고리즘1) + 겹치는 타일 병합(알고리즘2) 공용 모듈.
precompute_patches.py(학습용 patch 생성)와 evaluate_*.py(추론용 타일링) 양쪽에서 재사용.

[알고리즘 1] 손상 덩어리(connected component) 단위 crop
  - 기존: 마스크 전체의 global bounding box 기준으로 격자를 깔아서, 서로 떨어진
    손상 밴드가 있으면 그 사이 빈 공간까지 타일링하고 개별 밴드가 패치 경계에서
    잘리는 경우가 생겼음.
  - 개선: scipy.ndimage.label로 손상 영역을 개별 덩어리(component)로 분리하고,
    각 덩어리를 패치 중앙에 최대한 온전히 담기도록 crop. 덩어리가 patch_size보다
    크면 그 덩어리 내부에서만 겹치는 슬라이딩 윈도우로 커버.

[알고리즘 2] 겹치는 타일 병합 전략
  - 'average': 신뢰도(패치 중심에 가까울수록 높음) 가중 평균
  - 'best_tile': 픽셀마다 신뢰도가 가장 높은 타일 값 하나만 선택(경성 선택)
  신뢰도 = 패치 경계로부터의 거리 기반. 패치 중앙일수록 주변 문맥을 더 많이 보고
  예측했으므로 신뢰도가 높다는 원칙(타일 추론의 표준적인 접근).
"""

import numpy as np
from scipy.ndimage import label, find_objects


def get_damage_components(mask2d):
    """
    마스크 안의 서로 떨어진 손상 영역을 개별 덩어리로 분리.
    8-연결(대각선 포함)로 인접한 손상 픽셀을 하나의 덩어리로 묶음.
    반환: [(r0, r1, c0, c1), ...] 덩어리별 bounding box 리스트
    """
    structure = np.ones((3, 3))  # 8-연결
    labeled, n = label(mask2d > 0, structure=structure)
    if n == 0:
        return []
    objects = find_objects(labeled)
    boxes = []
    for sl in objects:
        if sl is None:
            continue
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        boxes.append((r0, r1, c0, c1))
    return boxes


def _gen_starts(lo, hi, size, stride, max_val):
    """[lo, hi) 구간을 patch_size 윈도우로 겹치며 덮는 시작점 목록 (evaluate_*.py와 동일 로직)."""
    if hi - lo <= size:
        # 덩어리가 패치보다 작으면: 덩어리를 패치 중앙에 오도록 1개만 배치 (알고리즘1 핵심)
        start = int(np.clip((lo + hi) // 2 - size // 2, 0, max_val - size))
        return [start]
    starts = list(range(lo, hi - size + 1, stride))
    if not starts or starts[-1] != hi - size:
        starts.append(hi - size)
    return [int(np.clip(s, 0, max_val - size)) for s in starts]


def generate_component_tiles(img, mask2d, patch_size=256, stride=64):
    """
    [알고리즘 1] 마스크의 각 손상 덩어리를 개별적으로, 패치 중앙에 최대한
    온전히 담기도록 crop한 (img_patch, mask_patch) 리스트 반환.
    같은 위치가 여러 덩어리에서 중복 생성될 수 있어 좌표 기준으로 중복 제거.
    """
    h, w = img.shape
    boxes = get_damage_components(mask2d)
    if not boxes:
        return []

    seen = set()
    tiles = []
    for (r0, r1, c0, c1) in boxes:
        row_starts = _gen_starts(r0, r1, patch_size, stride, h)
        col_starts = _gen_starts(c0, c1, patch_size, stride, w)
        for rs in row_starts:
            for cs in col_starts:
                key = (rs, cs)
                if key in seen:
                    continue
                mask_patch = mask2d[rs:rs + patch_size, cs:cs + patch_size]
                if mask_patch.sum() == 0:
                    continue
                seen.add(key)
                img_patch = img[rs:rs + patch_size, cs:cs + patch_size]
                tiles.append((rs, cs, img_patch, mask_patch))
    return tiles


def tile_confidence_map(mask_patch, patch_size=256):
    """
    [알고리즘 2용] 패치 내 각 픽셀의 신뢰도 맵.
    패치 중심에 가까울수록(=주변 문맥을 더 많이 보고 예측했으므로) 신뢰도가 높다는
    원칙으로, 패치 경계로부터의 최소 거리를 신뢰도로 사용 (0~1 정규화).
    마스킹 안 된(mask=0) 픽셀은 애초에 병합 대상이 아니므로 0 처리.
    """
    ps = patch_size
    yy, xx = np.mgrid[0:ps, 0:ps]
    dist_to_edge = np.minimum.reduce([yy, ps - 1 - yy, xx, ps - 1 - xx]).astype(np.float32)
    conf = dist_to_edge / (ps / 2)  # 중심=1.0에 가깝고 가장자리=0에 가까움
    conf = np.clip(conf, 0, 1)
    return conf * (mask_patch > 0)


def merge_tiles(h, w, tile_results, merge_mode="average"):
    """
    [알고리즘 2] 겹치는 타일들의 예측값을 병합.

    tile_results: [(rs, cs, pred_patch, mask_patch), ...] — 각 타일의 좌상단 좌표,
                  예측값(denormalize된 raw DN), 해당 타일의 마스크
    merge_mode:
      'average'   — 신뢰도 가중 평균 (여러 타일 예측을 부드럽게 블렌딩)
      'best_tile' — 픽셀마다 신뢰도가 가장 높은 타일 값 하나만 채택 (경성 선택)

    반환: (accum_result, weight_map) — weight_map>0인 곳이 실제로 채워진 영역
    """
    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    if merge_mode == "best_tile":
        best_conf = np.zeros((h, w), dtype=np.float32)

    for rs, cs, pred_patch, mask_patch in tile_results:
        ps_h, ps_w = mask_patch.shape
        conf = tile_confidence_map(mask_patch, patch_size=ps_h)

        if merge_mode == "average":
            accum[rs:rs + ps_h, cs:cs + ps_w] += pred_patch * conf
            weight[rs:rs + ps_h, cs:cs + ps_w] += conf
        elif merge_mode == "best_tile":
            region_conf = best_conf[rs:rs + ps_h, cs:cs + ps_w]
            better = conf > region_conf
            accum_region = accum[rs:rs + ps_h, cs:cs + ps_w]
            accum_region[better] = pred_patch[better]
            accum[rs:rs + ps_h, cs:cs + ps_w] = accum_region
            best_conf[rs:rs + ps_h, cs:cs + ps_w] = np.maximum(region_conf, conf)
            weight[rs:rs + ps_h, cs:cs + ps_w] = np.maximum(
                weight[rs:rs + ps_h, cs:cs + ps_w], (conf > 0).astype(np.float32))
        else:
            raise ValueError(f"알 수 없는 merge_mode: {merge_mode}")

    if merge_mode == "average":
        covered = weight > 0
        result = np.zeros((h, w), dtype=np.float32)
        result[covered] = accum[covered] / weight[covered]
        return result, covered
    else:  # best_tile
        covered = weight > 0
        return accum, covered