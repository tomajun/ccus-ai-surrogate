"""
make_dataset.py —— 批量生成训练数据集并保存为 .npz
==================================================
用法：
    python sim/make_dataset.py --n 800 --out data/synthetic.npz

输出 .npz 字段：
    fields : (N, T, grid_n, grid_n)  CO2 饱和度时空序列
    params : (N, 3)                  [injection_rate, caprock_perm, thickness]
    risks  : (N,)                    0/1/2 泄漏风险标签
"""
import argparse
import json
import os
import sys

# 保证仓库根目录在 sys.path（兼容 python sim/make_dataset.py 直接运行）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from sim.synthetic_field import generate_plume, risk_score


def load_priors(path: str = "data/priors.json"):
    """
    读取真实参数先验。文件不存在 / enabled 不为 true / 解析失败 -> 返回 None，
    此时 sample_params 回退到 uniform 均匀采样（不阻塞现有流程）。
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not data.get("enabled", False):
        return None
    return data.get("params", {})


def _sample_one(rng: np.random.Generator, spec: dict):
    """按单个参数的分布规格采样，支持 uniform / normal / lognormal / fixed。"""
    if not isinstance(spec, dict):
        return float(rng.uniform(0.5, 2.0))
    t = spec.get("type", "uniform")
    if t == "uniform":
        return float(rng.uniform(spec.get("low", 0.5), spec.get("high", 2.0)))
    if t == "normal":
        v = float(rng.normal(spec.get("mean", 1.0), spec.get("sigma", 0.3)))
        return float(np.clip(v, spec.get("low", -1e9), spec.get("high", 1e9)))
    if t == "lognormal":
        v = float(rng.lognormal(spec.get("mean", 0.0), spec.get("sigma", 0.5)))
        return float(np.clip(v, spec.get("low", 0.0), spec.get("high", 1e9)))
    if t == "fixed":
        return float(spec.get("value", 1.0))
    # 未知类型 -> 默认 uniform
    return float(rng.uniform(spec.get("low", 0.5), spec.get("high", 2.0)))


def sample_params(rng: np.random.Generator, priors: dict = None) -> dict:
    """
    采样一组注入参数。
    - 传入 priors（来自 data/priors.json 且 enabled=true）：按真实分布采样；
    - 否则：uniform 均匀采样（与历史行为一致，保证可复现）。
    """
    if priors:
        return dict(
            injection_rate=_sample_one(rng, priors.get("injection_rate", {"type": "uniform", "low": 0.5, "high": 2.0})),
            caprock_perm=_sample_one(rng, priors.get("caprock_perm", {"type": "uniform", "low": 0.1, "high": 1.0})),
            thickness=_sample_one(rng, priors.get("thickness", {"type": "uniform", "low": 20.0, "high": 40.0})),
        )
    return dict(
        injection_rate=float(rng.uniform(0.5, 2.0)),
        caprock_perm=float(rng.uniform(0.1, 1.0)),
        thickness=float(rng.uniform(20.0, 40.0)),
    )


def main(n: int = 800, out: str = "data/synthetic.npz", grid_n: int = 48, n_steps: int = 10):
    rng = np.random.default_rng(42)
    # 加载真实参数先验；不存在/未启用时返回 None -> 回退 uniform
    priors = load_priors(os.path.join(ROOT, "data", "priors.json"))
    if priors is not None:
        print("[info] 使用 data/priors.json 中的真实参数先验采样")
    else:
        print("[info] 未启用 priors.json，使用 uniform 均匀采样（默认）")
    fields, params_arr, risks = [], [], []
    for i in range(n):
        p = sample_params(rng, priors)
        f = generate_plume(p, grid_n, n_steps, seed=i)
        _, lab = risk_score(p, f[-1], grid_n)
        fields.append(f)
        params_arr.append([p["injection_rate"], p["caprock_perm"], p["thickness"]])
        risks.append(lab)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        fields=np.stack(fields).astype(np.float32),
        params=np.array(params_arr, dtype=np.float32),
        risks=np.array(risks, dtype=np.int64),
    )
    # 打印类别分布，方便检查是否平衡
    counts = np.bincount(np.array(risks), minlength=3)
    print(f"saved -> {out}")
    print(f"shape: fields={fields[0].shape}, total samples={n}")
    print(f"risk label distribution (low/mid/high): {counts.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", type=str, default="data/synthetic.npz")
    ap.add_argument("--grid_n", type=int, default=48)
    ap.add_argument("--n_steps", type=int, default=10)
    args = ap.parse_args()
    main(args.n, args.out, args.grid_n, args.n_steps)
