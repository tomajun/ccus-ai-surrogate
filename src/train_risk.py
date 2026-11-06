"""
train_risk.py —— 训练泄漏风险分级模型（进阶创新点 B）
=========================================================
用法：
    python -m src.train_risk --data data/synthetic.npz --epochs 30

输入：每样本 = 3 个注入参数 + 5 个预测场统计（见 risk_classifier.featurize）
输出：models/risk_mlp.pt  +  experiments/metrics_risk.json（Accuracy / Macro-F1）
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import RiskMLP, featurize


def build_xy(npz_path: str):
    d = np.load(npz_path)
    fields, params, risks = d["fields"], d["params"], d["risks"]
    X, Y = [], []
    for i in range(len(fields)):
        X.append(featurize(params[i], fields[i][-1]))
        Y.append(int(risks[i]))
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(Y, dtype=torch.long)


def train(data: str, epochs: int, batch: int, lr: float, device: str):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X, Y = build_xy(data)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = RiskMLP(in_dim=X.shape[1], n_cls=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"[risk] epoch {ep}/{epochs}  CE={run / len(ds):.4f}")

    # 评估
    model.eval()
    correct, n, preds, gts = 0, 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).argmax(1).cpu()
            correct += (out == yb).sum().item(); n += len(yb)
            preds.extend(out.tolist()); gts.extend(yb.tolist())
    from sklearn.metrics import classification_report
    acc = correct / n
    rep = classification_report(gts, preds, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    metrics = {"accuracy": round(acc, 4), "macro_f1": round(float(macro_f1), 4)}
    os.makedirs("models", exist_ok=True); os.makedirs("experiments", exist_ok=True)
    torch.save(model.state_dict(), "models/risk_mlp.pt")
    with open("experiments/metrics_risk.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("risk metrics:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args.data, args.epochs, args.batch, args.lr, args.device)
