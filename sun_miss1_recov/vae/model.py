"""
model.py — VAE 모델 (Deep + Dilated Bottleneck, MISS1 대응)

변경점:
1) 다운샘플링 4단계로 확장 (enc4/pool4 추가) → RF 68px → 140px
2) bottleneck_pre 뒤, to_mu/to_logvar 이전에 dilated conv block 추가
   → mu/logvar 자체가 넓은 문맥(RF 수백px)을 반영하도록 함
3) 나머지 구조/인터페이스(forward, vae_loss, kl_divergence, masked_l1_loss)는
   기존과 동일하게 유지 → train_vae.py 등 기존 학습 스크립트 수정 없이 그대로 사용 가능
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


def dilated_block(ch, dilation):
    """padding=dilation 으로 해상도 유지하면서 RF만 확장."""
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch

        # ---- Encoder (4단계로 확장) ----
        self.enc1 = conv_block(1, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Bottleneck: 채널 확장 + dilated conv로 mu/logvar 자체의 RF 확장 ----
        self.bottleneck_pre = conv_block(c * 8, c * 16)
        self.dilated1 = dilated_block(c * 16, dilation=2)
        self.dilated2 = dilated_block(c * 16, dilation=4)
        self.dilated3 = dilated_block(c * 16, dilation=8)

        self.to_mu = nn.Conv2d(c * 16, c * 16, 3, padding=1)
        self.to_logvar = nn.Conv2d(c * 16, c * 16, 3, padding=1)

        # ---- Decoder (4단계) ----
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)
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
        e4 = self.enc4(self.pool3(e3))

        h = self.bottleneck_pre(self.pool4(e4))
        h = self.dilated1(h)
        h = self.dilated2(h)
        h = self.dilated3(h)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar, (e1, e2, e3, e4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, skips):
        e1, e2, e3, e4 = skips
        d4 = self.up4(z)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
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
    diff = torch.abs(pred - target)
    weight = 1.0 + mask * (mask_weight - 1.0)
    return (diff * weight).mean()


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(pred, target, mask, mu, logvar, mask_weight=5.0, beta=0.1):
    recon = masked_l1_loss(pred, target, mask, mask_weight=mask_weight)
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl