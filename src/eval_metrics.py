"""
eval_metrics.py —— 加载已训好的权重，计算并保存指标（免重训）
============================================================
用途：训练脚本因故中断、或只想复算指标时，用本脚本直接加载
      models/<name>.pt 评估，无需重新训练（省算力）。

用法：
    python -m src.eval_metrics --ckpt models/surrogate_cnn.pt --model cnn
    python -m src.eval_metrics --ckpt models/surrogate_fno.pt --model fno
    python -m src.eval_metrics --ckpt models/risk_mlp.pt --model risk

输出：experiments/metrics_<model>.json （与 train.py 同格式）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import PlumeDataset
from src.models import UNetLite, FNO, RiskMLP, featurize
from src.train import ssim, _to_native


def _metrics_surrogate(ckpt, model_name, data, device):
    ds = PlumeDataset(data)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    model = (UNetLite() if model_name == "cnn" else FNO()).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
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
    x0 = torch.stack([ds[0][0]]).to(device)
    _ = model(x0)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = model(x0)
    lat = (time.perf_counter() - t0) / 50
    return {
        "model": model_name,
        "rmse": round(float(rmses / n), 4),
        "ssim": round(float(ssims / n), 4),
        "infer_latency_ms": round(float(lat * 1000), 3),
        "speedup_vs_sim_frame": round(float(500.0 / max(lat, 1e-4)), 1),
    }


def _metrics_risk(ckpt, data, device):
    d = np.load(data)
    fields, params, risks = d["fields"], d["params"], d["risks"]
    X, Y = [], []
    for i in range(len(fields)):
        X.append(featurize(params[i], fields[i][-1]))
        Y.append(int(risks[i]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    loader = DataLoader(list(zip(X, Y)), batch_size=32, shuffle=False)
    model = RiskMLP(in_dim=X.shape[1], n_cls=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    correct, n, preds, gts = 0, 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).argmax(1).cpu()
            correct += (out == yb).sum().item()
            n += len(yb)
            preds.extend(out.tolist())
            gts.extend(yb.tolist())
    from sklearn.metrics import classification_report
    acc = correct / n
    rep = classification_report(gts, preds, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="权重路径，如 models/surrogate_cnn.pt")
    ap.add_argument("--model", choices=["cnn", "fno", "risk"], required=True)
    ap.add_argument("--data", default="data/synthetic.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model == "risk":
        metrics = _metrics_risk(args.ckpt, args.data, device)
    else:
        metrics = _metrics_surrogate(args.ckpt, args.model, args.data, device)

    os.makedirs("experiments", exist_ok=True)
    out = f"experiments/metrics_{args.model}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=_to_native)
    print("metrics:", metrics, "->", out)


if __name__ == "__main__":
    main()
