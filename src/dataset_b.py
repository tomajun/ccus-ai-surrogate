"""
dataset_b.py —— 阶段 B 数据集封装（输入含异质渗透率场）
====================================================
把 .npz 变成模型能吃的张量。

阶段 B 的关键改动：模型输入从"3 个参数通道"扩展为"3 个参数通道 + 1 个
异质渗透率场通道"（共 4 通道）。这样模型才看得到地质结构，能学到
'渗透率场 -> 羽流形状' 的映射；否则同一组参数对应无数种羽流，异质差异
对模型只是噪声，UQ 永远解锁不了（阶段 A 的惨痛教训）。

    输入 X: 4 通道 (grid_n x grid_n)
        ch0 = injection_rate, ch1 = caprock_perm, ch2 = thickness (broadcast 常数)
        ch3 = perm_field  (异质渗透率相对场)  —— 阶段 B 新增
    目标 Y: 最终时刻(T-1)的 CO2 饱和度场 (grid_n x grid_n)
"""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PlumeDatasetB(Dataset):
    def __init__(self, npz_path: str, grid_n: int = 48):
        d = np.load(npz_path)
        self.fields = d["fields"].astype(np.float32)        # (N, T, H, W)
        self.params = d["params"].astype(np.float32)        # (N, 3)
        self.perm = d["perm_fields"].astype(np.float32)     # (N, H, W)
        self.grid_n = grid_n

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.fields[i]                                   # (T, H, W)
        p = self.params[i]                                   # (3,)
        perm = self.perm[i]                                  # (H, W)
        # 3 个参数广播成 (3, H, W) + perm 1 通道 -> 拼接为 4 通道输入
        p_ch = np.broadcast_to(p, (self.grid_n, self.grid_n, 3)).transpose(2, 0, 1)
        x = np.concatenate([p_ch, perm[None, ...]], axis=0)
        y = f[-1]                                            # 最终时刻场
        return torch.tensor(x), torch.tensor(y), torch.tensor(p)
