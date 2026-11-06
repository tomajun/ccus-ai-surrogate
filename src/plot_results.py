"""
plot_results.py —— 训练结果可视化（P1/P3 交付：预测对比图 + 混淆矩阵）
==============================================================
纯 matplotlib + numpy，不依赖 torch，沙箱/本机都能跑。

用法：
    python -m src.plot_results selftest                 # 用假数据验证出图管线
    python -m src.plot_results fields --ckpt models/surrogate_cnn.pt --out results/pred_compare.png
    python -m src.plot_results confusion --ckpt models/risk_mlp.pt --out results/confusion.png

说明：fields/confusion 子命令需要 torch 加载模型推理（本机跑）；selftest 只用 numpy 验证绘图。
"""
import argparse
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，安全用于服务器/沙箱
import matplotlib.pyplot as plt

# 尽量用中文字体（Windows 常见），找不到则退回默认；本机图内能正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_field_comparison(fields_true, fields_pred, out_path, n=4):
    """并排画 n 个样本的最终时刻场：左真值、右预测。"""
    n = min(n, len(fields_true))
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    for i in range(n):
        t = fields_true[i]
        p = fields_pred[i]
        ax0, ax1 = axes[i, 0], axes[i, 1]
        im0 = ax0.imshow(t, cmap="viridis", vmin=0, vmax=1)
        ax0.set_title(f"#{i} 真值"); ax0.axis("off")
        im1 = ax1.imshow(p, cmap="viridis", vmin=0, vmax=1)
        ax1.set_title(f"#{i} 预测"); ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confusion(y_true, y_pred, out_path, labels=("低", "中", "高")):
    """画混淆矩阵热力图 + 数值标注。"""
    n_cls = len(labels)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[int(a), int(b)] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(labels)
    ax.set_xlabel("预测"); ax.set_ylabel("真值")
    ax.set_title("风险分级混淆矩阵")
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _selftest():
    """用假数据验证绘图管线能出图（不依赖任何模型权重）。"""
    rng = np.random.default_rng(0)
    true = rng.random((4, 48, 48)).astype(np.float32)
    pred = np.clip(true * 0.9 + 0.04 * rng.standard_normal((4, 48, 48)), 0, 1).astype(np.float32)
    y_true = rng.integers(0, 3, size=60)
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=60), 0, 2)
    tmp = tempfile.mkdtemp(prefix="ccus_plot_")
    p1 = plot_field_comparison(true, pred, os.path.join(tmp, "pred_compare.png"), n=4)
    p2 = plot_confusion(y_true, y_pred, os.path.join(tmp, "confusion.png"))
    print(f"selftest 通过：\n  {p1}\n  {p2}\n（临时文件在 {tmp}，可忽略）")


def main():
    ap = argparse.ArgumentParser(description="训练结果可视化")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="用假数据验证出图（默认）")
    pf = sub.add_parser("fields", help="画真值 vs 预测对比图（需 torch）")
    pf.add_argument("--data", default="data/synthetic.npz")
    pf.add_argument("--ckpt", required=True)
    pf.add_argument("--model", choices=["cnn", "fno"], default="cnn")
    pf.add_argument("--out", default="results/pred_compare.png")
    pc = sub.add_parser("confusion", help="画风险混淆矩阵（需 torch）")
    pc.add_argument("--data", default="data/synthetic.npz")
    pc.add_argument("--ckpt", required=True)
    pc.add_argument("--out", default="results/confusion.png")
    args = ap.parse_args()
    cmd = getattr(args, "cmd", None) or "selftest"
    if cmd == "selftest":
        _selftest()
    elif cmd == "fields":
        import torch  # 延迟 import：避免沙箱无 torch 时 selftest 也挂
        from src.dataset import PlumeDataset
        from src.models import UNetLite, FNO
        ds = PlumeDataset(args.data)
        model = (UNetLite() if args.model == "cnn" else FNO())
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.eval()
        true, pred = [], []
        with torch.no_grad():
            for k in range(min(4, len(ds))):
                x, y, _ = ds[k]
                p = model(x.unsqueeze(0))[0, 0].numpy()
                true.append(y.numpy()); pred.append(p)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plot_field_comparison(np.stack(true), np.stack(pred), args.out, n=4)
        print(f"已保存预测对比图：{args.out}")
    elif cmd == "confusion":
        import torch
        from src.dataset import PlumeDataset
        from src.models import RiskMLP, featurize
        d = np.load(args.data)
        fields, params, risks = d["fields"], d["params"], d["risks"]
        X, Y = [], []
        for i in range(len(fields)):
            X.append(featurize(params[i], fields[i][-1])); Y.append(int(risks[i]))
        X = torch.tensor(np.stack(X), dtype=torch.float32)
        model = RiskMLP(in_dim=X.shape[1], n_cls=3)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(1).numpy()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plot_confusion(Y, preds, args.out)
        print(f"已保存混淆矩阵：{args.out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
