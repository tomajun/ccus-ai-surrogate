"""
dataset.py —— PyTorch 数据集封装
=================================
把 .npz 变成模型能吃的张量。

代理模型任务（基线/进阶统一口径，便于公平对比）：
    输入 X: 3 个"铺平为常数"的参数通道
           ch0 = injection_rate, ch1 = caprock_perm, ch2 = thickness
    目标 Y: 最终时刻(T-1)的 CO2 饱和度场 (grid_n x grid_n)，取值[0,1]

这样设计的好处：输入是"注入方案"，输出是"封存后的 CO2 分布"，
完美对应"给定注入参数，秒级预测封存结果"的代理建模叙事。
"""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PlumeDataset(Dataset):
    def __init__(self, npz_path: str, grid_n: int = 48):
        d = np.load(npz_path)
        self.fields = d["fields"].astype(np.float32)   # (N, T, N, N)
        self.params = d["params"].astype(np.float32)   # (N, 3)
        self.grid_n = grid_n

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.fields[i]                              # (T, N, N)
        p = self.params[i]                              # (3,)
        # 把 3 个参数广播成 (3, N, N) 的输入
        x = np.broadcast_to(p, (self.grid_n, self.grid_n, 3)).transpose(2, 0, 1)
        y = f[-1]                                       # 最终时刻场
        return torch.tensor(x), torch.tensor(y), torch.tensor(p)
