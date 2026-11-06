"""
cnn_surrogate.py —— 基线代理模型：轻量 U-Net（CNN）
=================================================
角色：把"注入参数"映射成"CO2 饱和度场"。
比喻：CNN 像一个有经验的绘图员，看到参数就照着画一张羽流分布图。
      它学的是"图像到图像"的映射，对空间结构捕捉不错，是稳的基线。

结构：编码(下采样提特征) -> 解码(上采样还原图) -> 1x1 卷积出单通道。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetLite(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        def conv(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(),
                nn.Conv2d(oc, oc, 3, padding=1), nn.ReLU(),
            )
        self.enc1 = conv(in_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv(32, 64)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = conv(64, 32)
        self.head = nn.Conv2d(32, out_ch, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        if u.shape != e1.shape:
            u = F.interpolate(u, size=e1.shape[-2:])
        d = self.dec(torch.cat([u, e1], dim=1))
        return self.sig(self.head(d))
