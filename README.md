# KASI SDO MISS 영상 복원 (AE / VAE / LaMa / Palette)

한국천문연구원(KASI) 우주탐사기술센터 하계 인턴십에서 진행한 프로젝트입니다.
SDO/AIA 131Å 태양 관측 영상에 발생하는 데이터 손실(MISS) 영역을 딥러닝으로 복원하고,
4가지 방법론(AE, VAE, LaMa, Palette Diffusion)의 성능을 비교 평가했습니다.

## 배경

SDO(Solar Dynamics Observatory) AIA 131Å 채널 영상은 readout channel 오류 등으로
가로 밴드 형태의 데이터 손실(MISS)이 발생합니다. 손상 규모에 따라 MISS0(협소 손상),
MISS1(중간 규모) 등으로 단계를 구분하고, 각 단계별로 복원 성능을 검증했습니다.

## 핵심 발견

픽셀 단위 지표(PSNR, L1)와 실제 시각적 품질이 항상 일치하지 않았습니다.
LaMa는 PSNR/L1 지표상으로는 AE/VAE보다 낮았지만, Fourier Convolution 기반의
넓은 receptive field 덕분에 실제 시각적으로는 더 자연스러운 복원 결과를 보였습니다.

| 모델 | masked_l1 | masked_rmse | masked_psnr(dB) | masked_ssim |
|---|---|---|---|---|
| AE | 3.82 | - | 47.32 | 0.8747 |
| VAE | 3.90 | - | 47.36 | 0.8757 |
| LaMa | 5.73 | - | 44.69 | 0.8424 |
| Palette* | 9.74 | 14.40 | 40.58 | 0.7084 |
| Baseline (linear) | 3.91 | - | 47.08 | 0.8527 |

*(MISS1 기준, 픽셀 지표는 AE/VAE가 우세하나 시각 품질은 LaMa가 더 자연스러움)*

\* Palette는 diffusion sample_steps=20 기준 잠정 결과입니다. 압축된 스텝 수로 인한
베타 스케줄 불일치로 시각적 노이즈(speckle)가 발생함을 확인했으며, sample_steps=200
기준 재평가 후 갱신 예정입니다.

## 복원 결과 비교

![comparison](sun_miss1_recov/verify_out/comparison_miss1/comparison_miss1_case1.png)

## 사용 모델

- **AE**: 4-level encoder + dilated bottleneck convolution 기반 오토인코더
- **VAE**: AE와 동일 구조에 확률적 잠재변수 추가
- **LaMa**: Fast Fourier Convolution 기반 GAN, 넓은 receptive field로 대형 손상에 강점
- **Palette**: Conditional Diffusion Model 기반 inpainting (DDPM 계열)

## 저장소 구조

    sun_miss0_recov/          # 협소 손상(MISS0) 파이프라인
    sun_miss1_recov/          # 중간 규모 손상(MISS1) 파이프라인
    ├── ae/ vae/ lama/ palette/   # 모델별 학습·평가 스크립트
    ├── baseline/              # 선형/3차 보간 baseline
    ├── mask_bank/             # 손상 위치 탐지 + 타일 병합 공용 모듈
    ├── make_comparison_missX.py  # 방법별 비교 이미지 생성
    └── verify_out/comparison_missX/  # 대표 복원 결과 이미지

## 기술 스택

Python, PyTorch, NumPy, Pandas, scipy.ndimage, matplotlib, astropy (FITS 처리)

## 참고

원본 학습 데이터(SDO FITS 파일)와 서버 경로는 보안상 제외되었으며,
경로 관련 상수는 `/path/to/...` 형태의 플레이스홀더로 대체되어 있습니다.
