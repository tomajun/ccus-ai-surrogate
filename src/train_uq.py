"""
train_uq.py —— 创新点①：不确定性量化（UQ，深度集成 Deep Ensemble）
================================================================
路径 A：用 K 个不同随机种子训 K 个代理模型，预测取均值，标准差即不确定性；
        对风险分级则训 K 个 RiskMLP，用预测概率的熵表示"模型自己也不确定"。

核心指标（写进申报书/答辩）：
    PICP : 真实值落在 90% 预测区间的比例（越接近 0.90 越好，区间要"准"）
    MPIW : 预测区间平均宽度（越窄越好，区间要"紧"）
    risk_high_entropy_ratio : 被判定为"高风险"的样本里，模型自身拿不准（高熵）的比例

用法：
    python -m src.train_uq --model cnn --seeds 5 --epochs 20
    python -m src.train_uq --model fno --seeds 5 --epochs 20
    python -m src.train_uq --risk --seeds 5 --epochs 30
输出：
    models/surrogate_<model>_uq_seed{0..K-1}.pt
    models/risk_mlp_uq_seed{0..K-1}.pt
    experiments/metrics_uq_<model>.json / experiments/metrics_uq_risk.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.dataset import PlumeDataset
from src.models import UNetLite, FNO, RiskMLP, featurize


def _to_native(o):
    """把 numpy 标量/数组转成 JSON 可序列化原生类型。"""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _train_surrogate_one(model, loader, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep}/{epochs} MSE={run / len(loader.dataset):.4f}")


def train_ensemble(model_name, data, seeds, epochs, batch, lr, alpha, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = PlumeDataset(data)
    os.makedirs("models", exist_ok=True)
    # 深度集成标准做法：每个成员在训练集上做 bootstrap 重采样（有放回），
    # 使各成员看到略不同的样本 -> 在样本稀疏区产生非平凡方差。否则在确定性
    # 数据上 K 个种子会收敛到同一函数，集成方差塌缩到 0（UQ 失效）。
    rng = np.random.default_rng(20240517)
    for s in seeds:
        torch.manual_seed(s)
        boot_idx = rng.choice(len(ds), size=len(ds), replace=True)
        loader = DataLoader(Subset(ds, boot_idx), batch_size=batch, shuffle=True)
        model = (UNetLite() if model_name == "cnn" else FNO()).to(device)
        print(f"[UQ/{model_name}] seed={s}")
        _train_surrogate_one(model, loader, epochs, lr, device)
        torch.save(model.state_dict(), f"models/surrogate_{model_name}_uq_seed{s}.pt")

    # 集成推理：加载 K 个模型，对全量数据预测
    models = []
    for s in seeds:
        m = (UNetLite() if model_name == "cnn" else FNO()).to(device)
        m.load_state_dict(torch.load(f"models/surrogate_{model_name}_uq_seed{s}.pt", map_location=device))
        m.eval(); models.append(m)
    all_x, all_y = [], []
    full_loader = DataLoader(ds, batch_size=batch, shuffle=False)
    for x, y, _ in full_loader:
        all_x.append(x); all_y.append(y)
    X_all = torch.cat(all_x, 0).to(device)
    Y_all = torch.cat(all_y, 0).numpy()
    with torch.no_grad():
        preds_K = torch.stack([m(X_all) for m in models], dim=0).cpu().numpy()  # (K,N,1,H,W)
    mean = preds_K.mean(axis=0)[..., 0]   # (N,H,W)
    std = preds_K.std(axis=0)[..., 0]
    z = 1.645 if abs(alpha - 0.90) < 1e-6 else 1.96
    lower, upper = mean - z * std, mean + z * std
    picp = float(((Y_all >= lower) & (Y_all <= upper)).mean())
    mpiw = float((2 * z * std).mean())
    metrics = {"model": model_name, "uq": "deep_ensemble", "seeds": list(seeds),
               "alpha": alpha, "picp": round(picp, 4), "mpiw": round(mpiw, 4)}
    os.makedirs("experiments", exist_ok=True)
    with open(f"experiments/metrics_uq_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_to_native)
    print("UQ metrics:", metrics)
    return metrics


def train_risk_uq(data, seeds, epochs, batch, lr, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    d = np.load(data)
    fields, params, risks = d["fields"], d["params"], d["risks"]
    X, Y = [], []
    for i in range(len(fields)):
        X.append(featurize(params[i], fields[i][-1])); Y.append(int(risks[i]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)
    os.makedirs("models", exist_ok=True)
    entropies = []
    for s in seeds:
        torch.manual_seed(s)
        model = RiskMLP(in_dim=X.shape[1], n_cls=3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = torch.nn.CrossEntropyLoss()
        for ep in range(1, epochs + 1):
            model.train(); run = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(model(xb), yb)
                loss.backward(); opt.step()
                run += loss.item() * xb.size(0)
        torch.save(model.state_dict(), f"models/risk_mlp_uq_seed{s}.pt")
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(X.to(device)), 1).cpu().numpy()
        entropies.append(-np.sum(p * np.log(p + 1e-9), axis=1))  # (N,)
    ent_mean = np.mean(entropies, axis=0)  # (N,)
    # 各 seed 预测取多数票
    preds_per_seed = []
    for s in seeds:
        m = RiskMLP(in_dim=X.shape[1], n_cls=3).to(device)
        m.load_state_dict(torch.load(f"models/risk_mlp_uq_seed{s}.pt", map_location=device)); m.eval()
        with torch.no_grad():
            preds_per_seed.append(m(X.to(device)).argmax(1).cpu().numpy())
    preds_per_seed = np.stack(preds_per_seed)  # (K,N)
    majority = np.array([int(np.bincount(preds_per_seed[:, i]).argmax()) for i in range(X.shape[0])])
    high_mask = (majority == 2)
    # 高熵阈值：均匀分布熵=log(3)≈1.0986，取一半 0.5493 表示"模型拿不准"
    high_uncert = float((ent_mean[high_mask] > 0.5493).mean()) if high_mask.sum() > 0 else 0.0
    metrics = {"uq": "deep_ensemble", "seeds": list(seeds),
               "mean_entropy": round(float(ent_mean.mean()), 4),
               "risk_high_entropy_ratio": round(high_uncert, 4)}
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/metrics_uq_risk.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_to_native)
    print("risk UQ metrics:", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="UQ 深度集成训练")
    ap.add_argument("--model", choices=["cnn", "fno"], default="cnn")
    ap.add_argument("--risk", action="store_true", help="做风险分级的 UQ 而非代理场 UQ")
    ap.add_argument("--data", default="data/synthetic.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.90, help="预测区间置信水平")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    if args.risk:
        train_risk_uq(args.data, seeds, args.epochs, args.batch, args.lr, args.device)
    else:
        train_ensemble(args.model, args.data, seeds, args.epochs, args.batch, args.lr,
                      args.alpha, args.device)


if __name__ == "__main__":
    main()
