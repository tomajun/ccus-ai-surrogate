"""
risk_classifier.py —— 泄漏风险智能分级（进阶创新点 B）
====================================================
角色：给定"注入参数 + 预测出的 CO2 场"，输出泄漏风险三档：低/中/高。
比喻：代理模型是"画羽流的人"，风险分类器是"看图的安检员"——
      它把羽流是否贴近盖层、是否大面积铺展、盖层渗透性如何，综合判成风险等级。

特征：3 个原始参数 + 5 个场统计（均值/标准差/铺展占比/最大值/顶部均值）。
模型：小巧 MLP，CPU 即可训练，输出 3 类 logits。
"""
import numpy as np
import torch
import torch.nn as nn


def featurize(params: np.ndarray, field: np.ndarray) -> np.ndarray:
    """把 (params, field) 变成定长特征向量。numpy 实现，便于推理与训练共用。"""
    s = field.reshape(-1)
    top = field[: max(1, field.shape[0] // 8), :].mean()
    stats = [
        float(s.mean()), float(s.std()),
        float((s > 0.2).mean()), float(s.max()), float(top),
    ]
    return np.concatenate([np.asarray(params, dtype=np.float32), np.asarray(stats, dtype=np.float32)])


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int = 8, n_cls: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_cls),
        )

    def forward(self, x):
        return self.net(x)
