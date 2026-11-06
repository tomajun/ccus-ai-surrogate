"""
fno_surrogate.py —— 进阶代理模型：傅里叶神经算子（Fourier Neural Operator）
==========================================================================
角色：同样把"注入参数"映射成"CO2 饱和度场"，但换了一种更聪明的学法。
比喻：CNN 是"照着画"；FNO 是"学会了物理规律本身"——
      它在频率域直接学算子，因此对新注入方案泛化更好、外推更稳，
      这正是论文里我们强调的"进阶创新点"。

实现要点：
  - SpectralConv2d：在 2D 傅里叶空间做线性变换（只保留低频模态 modes）。
  - FNO：参数投影 -> 多层谱卷积+非线性 -> 投影回单通道 -> Sigmoid。
注意：这是自包含的精简实现，便于本科生阅读；生产级可换 neuralop 库。
"""
import torch
import torch.nn as nn
import torch.fft as fft


class SpectralConv2d(nn.Module):
    """在傅里叶空间做线性算子（保留前 modes 个低频）。"""

    def __init__(self, in_ch: int, out_ch: int, modes: int = 12):
        super().__init__()
        self.modes = modes
        # 复数权重用 (real, imag) 两个通道存，最后 view_as_complex
        self.weight = nn.Parameter(torch.randn(in_ch, out_ch, modes, modes, 2) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x)
        out_ft = torch.zeros(B, self.weight.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        w = torch.view_as_complex(self.weight)
        for i in range(self.modes):
            for j in range(self.modes):
                out_ft[:, :, i, j] = torch.einsum("bc,co->bo",
                                                  x_ft[:, :, i, j], w[:, :, i, j])
        return fft.irfft2(out_ft, s=(H, W))


class FNO(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, modes: int = 12, width: int = 32):
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.sp1 = SpectralConv2d(width, width, modes)
        self.sp2 = SpectralConv2d(width, width, modes)
        self.conv = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Conv2d(width, 16, 1)
        self.fc2 = nn.Conv2d(16, out_ch, 1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.act(self.sp1(x))
        x = self.act(self.sp2(x))
        x = self.act(self.conv(x))
        x = self.act(self.fc1(x))
        return self.sig(self.fc2(x))
