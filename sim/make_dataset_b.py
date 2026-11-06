"""
make_dataset_b.py —— 阶段 B 批量数据集生成（异质场）
====================================================
用法：
    python sim/make_dataset_b.py --n 800 --out data/synthetic_b.npz

输出 .npz 字段（在阶段 A 基础上新增 perm_fields，作为模型输入）：
    fields      : (N, T, grid_n, grid_n)  CO2 饱和度时空序列（异质场演化）
    params      : (N, 3)                  [injection_rate, caprock_perm, thickness]
    risks       : (N,)                    0/1/2 泄漏风险标签（复用阶段 A 的 risk_score）
    perm_fields : (N, grid_n, grid_n)      异质渗透率相对场 —— 阶段 B 新增，将作为模型输入

说明：
    - 参数先验沿用 data/priors.json（enabled=true 时按真实分布采样，
      否则 uniform 回退）。
    - 每个样本独立生成一张异质渗透率场（seed=i），再用它演化羽流，
      因此样本间"地质结构"各异 —— 这是阶段 B 相对阶段 A 多出的最关键信号。
    - 阶段 A 的 synthetic.npz 保留不动，用于对照实验。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from sim.make_dataset import sample_params, load_priors
from sim.synthetic_field import risk_score
from sim.synthetic_field_b import generate_perm_field, generate_plume_hetero


def main(n: int = 800, out: str = "data/synthetic_b.npz",
         grid_n: int = 48, n_steps: int = 10):
    rng = np.random.default_rng(42)
    priors = load_priors(os.path.join(ROOT, "data", "priors.json"))
    if priors is not None:
        print("[info] 使用 data/priors.json 真实参数先验采样")
    else:
        print("[info] 未启用 priors.json，使用 uniform 均匀采样")

    fields, params_arr, risks, perms = [], [], [], []
    for i in range(n):
        p = sample_params(rng, priors)
        k = generate_perm_field(grid_n, rng, seed=i)
        f = generate_plume_hetero(p, k, grid_n, n_steps, seed=i)
        _, lab = risk_score(p, f[-1], grid_n)
        fields.append(f)
        params_arr.append([p["injection_rate"], p["caprock_perm"], p["thickness"]])
        risks.append(lab)
        perms.append(k)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        fields=np.stack(fields).astype(np.float32),
        params=np.array(params_arr, dtype=np.float32),
        risks=np.array(risks, dtype=np.int64),
        perm_fields=np.stack(perms).astype(np.float32),
    )
    counts = np.bincount(np.array(risks), minlength=3)
    print(f"saved -> {out}")
    print(f"shape: fields={fields[0].shape}, total samples={n}")
    print(f"risk label distribution (low/mid/high): {counts.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", type=str, default="data/synthetic_b.npz")
    ap.add_argument("--grid_n", type=int, default=48)
    ap.add_argument("--n_steps", type=int, default=10)
    args = ap.parse_args()
    main(args.n, args.out, args.grid_n, args.n_steps)
