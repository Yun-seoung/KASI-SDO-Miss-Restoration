"""
model.py — AE 모델 (이전 Tiny ImageNet 실습의 skip-connection U-Net을
1채널 태양 이미지(패치 256x256)에 맞게 재구성)
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


class AE(nn.Module):
    """
    U-Net 스타일 AE (skip connection 포함).
    입력: (B,1,256,256) 손상+정규화 이미지
    출력: (B,1,256,256) 복원 이미지 (sigmoid로 [0,1] 범위 보장)
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

        self.bottleneck = conv_block(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = torch.sigmoid(self.out_conv(d1))
        return out


def masked_l1_loss(pred, target, mask, mask_weight=5.0):
    """
    마스킹 영역에 mask_weight배 가중치를 준 L1 loss.
    (이전 AE/VAE 실습 때와 동일한 방식: mask_weight=5.0)
    """
    diff = torch.abs(pred - target)
    weight = 1.0 + mask * (mask_weight - 1.0)
    return (diff * weight).mean()