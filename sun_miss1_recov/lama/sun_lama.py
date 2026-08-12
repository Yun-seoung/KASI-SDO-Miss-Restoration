"""
sun_lama.py — Lama 모델을 만들고, 그걸 학습시키기 위한 손실 함수들을 정의한 도구 모음.

[재학습 개선 — 원인 분석 결과 반영]
기존 220epoch 학습 로그 분석 결과, adv/fm loss가 학습 시작부터 끝까지 거의
개선되지 않고 박스권에 갇혀 있었고(d도 ln(2)=0.693 근처에서 못 벗어남),
그 결과 큰 손상 영역에서 LaMa가 텍스처 없이 평평한 평균값으로 수렴하는
문제가 시각적으로 확인됨. 원인 조사 결과:

  [원인 A] NLayerDiscriminator의 norm_layer가 기본값 BatchNorm2d로 고정되어
           있었음. batch=4처럼 작은 배치에서는 BatchNorm 통계(평균/분산)가
           배치마다 크게 흔들려 D의 판별 신호가 노이즈에 오염되고, G가
           안정적인 adv/fm gradient를 못 받아 "안전한 평균값" 출력으로
           수렴하는 원인이 됨.
           -> InstanceNorm2d로 교체 (배치 크기에 의존하지 않는 정규화).

  [원인 B] w_fm=100.0이 w_adv=10.0보다 훨씬 커서, 학습 초반 아직 덜 여문
           D의 중간 feature를 강하게 따라가도록 G를 몰아붙였고, 이로 인해
           나쁜 지역해에 조기 정착했을 가능성.
           -> warm-up 스케줄 도입: 초반 N epoch은 adv=fm=0(순수 l1만 학습)
              으로 G가 구조를 먼저 잡게 한 뒤, 이후 M epoch에 걸쳐 목표
              가중치까지 선형 증가(train_lama.py의 set_progress()에서 매
              epoch 갱신).

기존 인터페이스(build_generator, make_lama_input, make_eval_noise 등)는
evaluate_lama.py와 호환되도록 그대로 유지.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from saicinpainting.training.modules.ffc import FFCResNetGenerator
    from saicinpainting.training.modules.pix2pixhd import NLayerDiscriminator
except ImportError as e:
    raise ImportError(
        "LaMa 모듈 import 실패. LaMa 저장소 루트를 PYTHONPATH에 추가하세요.\n"
        "  export PYTHONPATH=/path/to/lama:$PYTHONPATH\n"
        f"  (원본 오류: {e})"
    )


# ---------------------------------------------------------------------------
# 모델 빌더 -> 복원 모델(G) 만들기
# ---------------------------------------------------------------------------
def build_generator(img_ch: int, mask_ch: int = 1, n_blocks: int = 9) -> nn.Module:
    """FFCResNetGenerator. input_nc=img+mask, output_nc=img."""
    return FFCResNetGenerator(
        input_nc=img_ch + mask_ch,
        output_nc=img_ch,
        ngf=64,
        n_downsampling=3,
        n_blocks=n_blocks,
        init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        resnet_conv_kwargs=dict(ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
        add_out_act="sigmoid",
    )


def build_discriminator(img_ch: int) -> nn.Module:
    # [원인 A 대응] 기본 BatchNorm2d는 batch=4처럼 작은 배치에서 통계가 불안정해져
    # D의 판별 신호가 노이즈에 흔들리고, 그 결과 G가 adv/fm 신호를 못 따라가며
    # 학습 초반부터 "안전한 평균값" 출력으로 수렴하는 원인이 됨.
    # InstanceNorm2d는 샘플 단위로 정규화되어 배치 크기에 의존하지 않음.
    return NLayerDiscriminator(input_nc=img_ch, ndf=64, n_layers=4,
                                norm_layer=nn.InstanceNorm2d)


# ---------------------------------------------------------------------------
# 손실
# ---------------------------------------------------------------------------
def masked_l1(pred, target, mask, w_known=10.0, w_missing=0.0):
    per_pix = F.l1_loss(pred, target, reduction="none")
    n_known = (1.0 - mask).sum().clamp_min(1.0) # mask=0(정상) 픽셀 개수
    n_missing = mask.sum().clamp_min(1.0)       # mask=1(손상) 픽셀 개수
    l_known = (per_pix * (1.0 - mask)).sum() / n_known
    l_missing = (per_pix * mask).sum() / n_missing
    return w_known * l_known + w_missing * l_missing


def feature_matching_loss(fake_feats, real_feats):
    if not fake_feats:
        return torch.zeros((), device=fake_feats[0].device if fake_feats else "cpu")
    loss = 0.0
    for f, r in zip(fake_feats, real_feats):
        loss = loss + F.l1_loss(f, r.detach())
    return loss / len(fake_feats)


def r1_penalty(real_logits, real_input):
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


class LamaLosses:
    def __init__(self, w_l1_known=10.0, w_adv_max=10.0, w_fm_max=30.0, r1_coef=0.001):
        self.w_l1_known = w_l1_known
        self.w_adv_max = w_adv_max
        self.w_fm_max = w_fm_max
        self.r1_coef = r1_coef
        # [원인 B 대응] 현재 스텝에서 실제로 적용할 가중치. train_lama.py가
        # set_progress()로 매 epoch 갱신함. 초기값은 0 (warm-up 시작 = l1만 학습).
        self.w_adv = 0.0
        self.w_fm = 0.0

    def set_progress(self, progress: float):
        """progress: 0.0(warm-up 시작, adv/fm=0) ~ 1.0(목표 가중치 전부 적용).
        adv/fm을 처음부터 최대 가중치로 걸지 않고 서서히 ramp-up해서 학습 초반에
        fm(가중치 100)이 아직 덜 여문 D의 feature를 강하게 따라가며 나쁜 지역해에
        갇히는 것을 방지."""
        progress = max(0.0, min(1.0, progress))
        self.w_adv = self.w_adv_max * progress
        self.w_fm = self.w_fm_max * progress

    def d_loss(self, D, real_img, fake_img, mask):
        """
        기존 버그 수정 유지: fake_target을 마스크 기반으로 만듦.
          - 마스킹 안 된 영역(원본과 같아야 하는 부분) -> 1 (진짜)
          - 마스킹된 영역(모델이 채워 넣은 부분) -> 0 (가짜)
        real_img는 전체가 진짜이므로 real_target은 그대로 전부 1.
        """
        real_img = real_img.detach().requires_grad_(True)
        real_logit, _ = D(real_img)
        fake_logit, _ = D(fake_img.detach())

        real_target = torch.ones_like(real_logit)

        mask_ds = F.interpolate(mask, size=fake_logit.shape[-2:], mode="nearest")
        fake_target = (1.0 - mask_ds).expand_as(fake_logit)

        d_real = F.binary_cross_entropy_with_logits(
            real_logit.flatten(), real_target.flatten()
        )
        d_fake = F.binary_cross_entropy_with_logits(
            fake_logit.flatten(), fake_target.flatten()
        )
        r1 = r1_penalty(real_logit, real_img)
        return d_real + d_fake + self.r1_coef * r1

    def g_loss(self, D, pred_img, real_img, mask):
        l1 = masked_l1(pred_img, real_img, mask, w_known=self.w_l1_known, w_missing=0.0)
        fake_logit, fake_feats = D(pred_img)
        with torch.no_grad():
            _, real_feats = D(real_img)
        adv = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
        fm = feature_matching_loss(fake_feats, real_feats)
        # [원인 B 대응] 고정 w_adv/w_fm 대신 warm-up 진행도에 따라 갱신된 값 사용
        total = l1 + self.w_adv * adv + self.w_fm * fm
        return total, {"l1": l1.item(), "adv": adv.item(), "fm": fm.item()}


# ---------------------------------------------------------------------------
# 노이즈
# ---------------------------------------------------------------------------
def make_lama_input(img, mask, noise: torch.Tensor = None, noise_std: float = 0.0):
    """LaMa 규약: concat([img*(1-mask), mask], dim=1).
    noise가 주어지면 그 텐서를 그대로 사용(결정론적 평가용).
    noise가 없고 noise_std>0이면 그 자리에서 순수 랜덤 생성(학습용, 매번 다름)."""
    if noise is not None:
        masked = torch.where(mask.bool(), noise.to(img.device), img)
    elif noise_std > 0:
        noise = torch.randn_like(img) * noise_std
        masked = torch.where(mask.bool(), noise, img)
    else:
        masked = img * (1.0 - mask)

    return torch.cat([masked, mask], dim=1)


def make_eval_noise(shape, index: int, seed: int, noise_std: float = 0.05) -> torch.Tensor:
    """평가 전용 결정론적 노이즈. mask 생성과 동일한 (index,seed) 패턴이지만
    RNG 스트림이 섞이지 않도록 mask와 다른 배수(7_000_003)를 씀.
    shape: (C,H,W) — 배치 없는 단일 샘플 형태."""
    rng = np.random.RandomState(seed * 7_000_003 + index)
    noise = rng.randn(*shape).astype(np.float32) * noise_std
    return torch.from_numpy(noise)


if __name__ == "__main__":
    for name, ch in [("A(color)", 3), ("B(gray)", 1)]:
        G = build_generator(ch)
        D = build_discriminator(ch)
        img = torch.rand(2, ch, 64, 64)
        mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
        inp = make_lama_input(img, mask, noise_std=0.05)
        out = G(inp)
        logit, feats = D(out)
        assert inp.shape[1] == ch + 1 and out.shape[1] == ch
        print(f"[{name}] G_in={inp.shape[1]}ch out={out.shape[1]}ch "
              f"range=({out.min().item():.3f},{out.max().item():.3f}) "
              f"D_logit={tuple(logit.shape)} n_feats={len(feats)} OK")