"""
model.py — VAE 모델
(sun_miss0_recov/ae/model.py의 U-Net skip-connection AE와 인코더/디코더/스킵 구조를
 완전히 동일하게 유지하고, bottleneck 자리만 mu/logvar 두 갈래로 분기)

핵심 차이점 (AE 대비):
  - bottleneck: conv_block(c*4, c*8) 그대로 통과시킨 뒤,
                to_mu / to_logvar 두 개의 conv로 분기
  - reparameterize: z = mu + std * eps  (std = exp(0.5*logvar))
  - decoder에는 z가 들어감 (AE의 b 자리와 동일 shape)
  - loss = masked_l1_loss(AE와 동일) + beta * KL divergence

주의: eval 모드(model.eval())에서는 랜덤 샘플링 없이 mu를 그대로 사용
      (평가 재현성을 위해 결정적으로 동작하도록 설계)
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    """
    U-Net 스타일 VAE (skip connection 포함, AE와 동일한 인코더/디코더).
    입력: (B,1,256,256) 손상+정규화 이미지
    출력: (B,1,256,256) 복원 이미지 (sigmoid로 [0,1] 범위 보장), mu, logvar
    """

    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc1 = conv_block(1, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)

        # --- AE의 self.bottleneck 자리: conv_block은 그대로 두고 mu/logvar로 분기 ---
        self.bottleneck_pre = conv_block(c * 4, c * 8)
        self.to_mu = nn.Conv2d(c * 8, c * 8, 3, padding=1)
        self.to_logvar = nn.Conv2d(c * 8, c * 8, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, 1, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        h = self.bottleneck_pre(self.pool3(e3))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar, (e1, e2, e3)

    def reparameterize(self, mu, logvar):
        # z = mu + std * eps  (eps ~ N(0,1))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, skips):
        e1, e2, e3 = skips
        d3 = self.up3(z)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = torch.sigmoid(self.out_conv(d1))
        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z, skips)
        return recon, mu, logvar


def masked_l1_loss(pred, target, mask, mask_weight=5.0):
    """AE와 완전히 동일한 정의 (재현성 확보용, ae/model.py에서 그대로 복사)."""
    diff = torch.abs(pred - target)
    weight = 1.0 + mask * (mask_weight - 1.0)
    return (diff * weight).mean()


def kl_divergence(mu, logvar):
    """
    표준정규분포 N(0,1)과의 KL divergence (spatial latent 기준 원소별 평균).
    KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(pred, target, mask, mu, logvar, mask_weight=5.0, beta=0.1):
    """전체 loss = masked_l1(recon) + beta * KL"""
    recon = masked_l1_loss(pred, target, mask, mask_weight=mask_weight)
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl