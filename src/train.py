"""
train.py —— 训练代理模型并评估（基线 CNN / 进阶 FNO）
====================================================
用法：
    python -m src.train --model cnn --epochs 20 --data data/synthetic.npz
    python -m src.train --model fno --epochs 20

输出：
    models/surrogate_<model>.pt
    experiments/metrics_<model>.json   (RMSE / SSIM / 推理延迟 / vs 仿真加速比)

指标说明：
    RMSE  : 预测场与真值场逐像素误差（越小越好）
    SSIM  : 结构相似度（越接近 1 越好，比 RMSE 更贴近"肉眼像不像"）
    speedup: 我们把"数值仿真一步"类比成一帧渲染开销，汇报代理模型单样本推理延迟，
            以及相对"逐帧仿真"的理论加速比（论文里可标注为概念性加速比）。
"""
import argparse
import json
import os
import sys
import time

# 保证仓库根目录在 sys.path（兼容 python src/train.py 与 python -m src.train 两种启动方式）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PlumeDataset
from src.models import UNetLite, FNO


def ssim(a: np.ndarray, b: np.ndarray, data_range=1.0) -> float:
    """极简 SSIM（单通道 2D），用于评估羽流形状相似度。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2)))


def _to_native(o):
    """把 numpy 标量/数组转成 JSON 可序列化原生类型。"""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def train(model_name: str, data: str, epochs: int, batch: int, lr: float, device: str):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = PlumeDataset(data)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = (UNetLite() if model_name == "cnn" else FNO()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
        if ep % 5 == 0 or ep == 1:
            print(f"[{model_name}] epoch {ep}/{epochs}  MSE={run / len(ds):.4f}")

    # 评估
    model.eval()
    rmses, ssims, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()[0, 0]
            gt = y.numpy()[0]
            rmses += np.sqrt(((pred - gt) ** 2).mean())
            ssims += ssim(pred, gt)
            n += 1
    # 推理延迟（单样本）
    x0 = torch.stack([ds[0][0]]).to(device)
    _ = model(x0)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = model(x0)
    lat = (time.perf_counter() - t0) / 50

    metrics = {
        "model": model_name,
        # 全部显式转原生 float/int，避免 numpy 标量（float32/64）在 json.dump 时报
        # "not JSON serializable"——此前 FNO 训练即卡在此处。
        "rmse": round(float(rmses / n), 4),
        "ssim": round(float(ssims / n), 4),
        "infer_latency_ms": round(float(lat * 1000), 3),
        "speedup_vs_sim_frame": round(float(500.0 / max(lat, 1e-4)), 1),  # 概念性：仿真一帧估 0.5s
    }
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    torch.save(model.state_dict(), f"models/surrogate_{model_name}.pt")
    with open(f"experiments/metrics_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_to_native)
    print("metrics:", metrics)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "fno"], default="cnn")
    ap.add_argument("--data", default="data/synthetic.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args.model, args.data, args.epochs, args.batch, args.lr, args.device)
