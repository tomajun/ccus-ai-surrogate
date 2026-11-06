"""
train_b.py —— 阶段 B 训练 + 深度集成 UQ 验证（轻量）
===================================================
目的：验证"把异质渗透率场作为输入"是否真的能解锁 UQ。
      对照：阶段 A 深度集成 UQ 在玩具数据上崩溃（PICP 仅 0.7%~24%、MPIW≈0）。

做法：对同一份阶段 B 数据，用 K 个随机种子各训一个模型，验证集预测取
      均值 ± 2σ 作为 95% 置信区间，计算 PICP（区间覆盖率，目标≈0.95）
      与 MPIW（平均区间宽度，越大代表不确定性估计越有信息量）。

用法：
    python src/train_b.py --data data/synthetic_b.npz --model cnn --seeds 5 --epochs 15
    python src/train_b.py --data data/synthetic_b.npz --model fno --seeds 5 --epochs 15
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset_b import PlumeDatasetB
from src.models.cnn_surrogate import UNetLite
from src.models.fno_surrogate import FNO


def _to_native(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_native(v) for v in o]
    return o


def train_one(seed: int, model_name: str, data: str, epochs: int, batch: int, lr: float):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = PlumeDatasetB(data)
    n = len(ds)
    n_val = max(1, n // 5)
    val_idx = list(range(n - n_val, n))
    tr_idx = list(range(n - n_val))
    # 深度集成标准做法：每个成员在训练集上做 bootstrap 重采样（有放回），
    # 使各成员看到略不同的样本 -> 在样本稀疏区产生非平凡方差。否则在确定性
    # 数据上 K 个种子会收敛到同一函数，集成方差塌缩到 0（UQ 失效）。
    rng_b = np.random.default_rng(seed)
    boot_idx = rng_b.choice(tr_idx, size=len(tr_idx), replace=True)
    tr = DataLoader(Subset(ds, boot_idx), batch_size=batch, shuffle=True)
    va = DataLoader(Subset(ds, val_idx), batch_size=batch)

    model = UNetLite(in_ch=4) if model_name == "cnn" else FNO(in_ch=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb, _ in tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds, yt = [], []
    with torch.no_grad():
        for xb, yb, _ in va:
            preds.append(model(xb))
            yt.append(yb)
    return torch.cat(preds).cpu().numpy(), torch.cat(yt).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_b.npz")
    ap.add_argument("--model", default="cnn", choices=["cnn", "fno"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_pred, all_y = [], None
    for s in range(args.seeds):
        p, y = train_one(s, args.model, args.data, args.epochs, args.batch, args.lr)
        all_pred.append(p)
        all_y = y
    all_pred = np.stack(all_pred)                  # (seeds, N_val, H, W)
    mean = all_pred.mean(0)
    std = all_pred.std(0)

    # 95% 名义区间：mean ± 2σ
    lo, hi = mean - 2 * std, mean + 2 * std
    picp = float(((all_y >= lo) & (all_y <= hi)).mean())
    mpiw = float((2 * 2 * std).mean())

    metrics = {
        "model": args.model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "picp_95": picp,                 # 目标 ≈ 0.95；阶段 A 曾崩溃至 0.007~0.24
        "mpiw": mpiw,                    # 区间平均宽度，越大代表 UQ 越有信息量
        "mean_std": float(std.mean()),    # 集成的平均预测不确定性
        "note": "阶段B深度集成UQ验证：对比阶段A PICP 0.7%~24% 的崩溃",
    }
    out = args.out or f"experiments/metrics_uq_b_{args.model}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_to_native(metrics), fh, indent=2)
    print(json.dumps(_to_native(metrics), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
