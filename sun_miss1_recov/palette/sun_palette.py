"""
sun_palette.py — 원본 Palette 저장소의 Network(모델 로직) + mse_loss + make_beta_schedule을
코드 수정 없이 그대로 import해서 빌더 함수로 감싼 것.
(lama/sun_lama.py가 원본 FFCResNetGenerator를 그대로 import했던 것과 동일한 패턴)

PYTHONPATH에 Palette 저장소 루트가 잡혀 있어야 함:
  export PYTHONPATH=/path/to/Palette-Image-to-Image-Diffusion-Models:$PYTHONPATH
"""

try:
    from models.network import Network  # 원본, 수정 없음
    from models.loss import mse_loss     # 원본, 수정 없음
except ImportError as e:
    raise ImportError(
        "Palette 모듈 import 실패. Palette 저장소 루트를 PYTHONPATH에 추가하세요.\n"
        "  export PYTHONPATH=/path/to/"
        "Palette-Image-to-Image-Diffusion-Models:$PYTHONPATH\n"
        f"  (원본 오류: {e})"
    )

# 원본 config/inpainting_tiny_full.json의 unet 설정을 그대로 가져오되,
# image_size만 64(Tiny ImageNet) -> 256(SDO patch)로 변경.
UNET_CONFIG = dict(
    in_channel=2,
    out_channel=1,
    inner_channel=64,
    channel_mults=[1, 2, 4, 8],
    attn_res=[16],
    num_head_channels=32,
    res_blocks=2,
    dropout=0.2,
    image_size=512,
)

BETA_SCHEDULE = {
    "train": {"schedule": "linear", "n_timestep": 1000,
              "linear_start": 1e-06, "linear_end": 0.01},
    "test": {"schedule": "linear", "n_timestep": 200,
             "linear_start": 0.0001, "linear_end": 0.09},
}


def build_palette_network(device="cuda"):
    """
    원본 Network를 guided_diffusion U-Net + linear beta schedule로 빌드.
    module_name='guided_diffusion'은 config/inpainting_tiny_full.json에서 확인된 원본 설정.
    """
    net = Network(unet=UNET_CONFIG, beta_schedule=BETA_SCHEDULE, module_name="guided_diffusion")
    net.set_loss(mse_loss)
    net.set_new_noise_schedule(device=device, phase="train")
    net.to(device)
    return net


def set_eval_noise_schedule(net, device="cuda", n_timestep=None):
    """평가(샘플링) 시에는 test용 스케줄로 다시 세팅. n_timestep을 주면 덮어씀."""
    if n_timestep is not None:
        net.beta_schedule["test"]["n_timestep"] = n_timestep
    net.set_new_noise_schedule(device=device, phase="test")