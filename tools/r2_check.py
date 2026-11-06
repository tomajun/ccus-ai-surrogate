# 测量申报书承诺的 R²（掩膜内，点预测口径）。
# 用法：.venv/Scripts/python.exe tools/r2_check.py --limit 100
# 输出：experiments/ufno_r2_check.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro", "ufno"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _expose_to_main():
    """官方权重是 notebook 里保存的整模型 pickle，类名被记成 __main__.Net3d。"""
    import __main__
    import ufno as _u
    import lploss as _l
    for _mod in (_u, _l):
        for _name in dir(_mod):
            if isinstance(getattr(_mod, _name), type):
                setattr(__main__, _name, getattr(_mod, _name))


def masked_r2(pred, true, mask):
    sse = ((pred - true) ** 2)[mask].sum()
    sst = ((true - true[mask].mean()) ** 2)[mask].sum()
    return float(1 - sse / sst)


def main():
    ap = argparse.ArgumentParser(description="Masked R^2 of the U-FNO point prediction")
    ap.add_argument("--model_path", default="数据集/sg_UFNO_4500_trained.pt")
    ap.add_argument("--data_dir", default="数据集")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100, help="场地数（0=全部 500）")
    ap.add_argument("--sat_thr", type=float, default=0.01, help="羽流区阈值，与 run_uq_risk 同口径")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="experiments/ufno_r2_check.json")
    args = ap.parse_args()

    from ufno_reader import open_split  # noqa: E402

    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _expose_to_main()
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.eval()

    a, y = open_split(args.data_dir, args.split)
    n = a.shape[0] if args.limit <= 0 else min(args.limit, a.shape[0])
    r2_all, r2_plume, rels = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            x = torch.from_numpy(np.array(a[i:i + 1])).to(device)
            yy = torch.from_numpy(np.array(y[i:i + 1])).to(device)
            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)
            pred = model(x).view(-1, 96, 200, 24)
            m, p, t = mask[0], pred[0], yy[0]
            r2_all.append(masked_r2(p, t, m))
            pl = (t > args.sat_thr) & m
            if int(pl.sum()) > 100:
                r2_plume.append(masked_r2(p, t, pl))
            rels.append(float((p - t)[m].norm() / t[m].norm()))

    arr, pl_arr, rel = np.array(r2_all), np.array(r2_plume), np.array(rels)
    out = {
        "model_path": args.model_path,
        "split": args.split,
        "n_samples": int(n),
        "sat_thr": args.sat_thr,
        "device": str(device),
        "note": "点预测（单次前向），非 MC 均值；掩膜内计算，与 eval 脚本同口径",
        "r2_masked": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "min": float(arr.min()),
            "share_above_0_99": float((arr > 0.99).mean()),
            "share_above_0_995": float((arr > 0.995).mean()),
        },
        "r2_plume_region": {
            "n_sites": int(len(pl_arr)),
            "mean": float(pl_arr.mean()),
            "min": float(pl_arr.min()),
            "share_above_0_99": float((pl_arr > 0.99).mean()),
        },
        "rel_l2_masked": {
            "mean": float(rel.mean()),
            "p90": float(np.percentile(rel, 90)),
            "max": float(rel.max()),
        },
        "per_site_r2": [float(v) for v in arr],
        "seconds_per_sample": float((time.time() - t0) / n),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"[r2] N={n} device={device}")
    print("  masked R2   mean %.5f median %.5f p10 %.5f min %.5f" %
          (arr.mean(), np.median(arr), np.percentile(arr, 10), arr.min()))
    print("  share >0.99 %.1f%%   >0.995 %.1f%%" %
          (100 * (arr > 0.99).mean(), 100 * (arr > 0.995).mean()))
    print("  plume R2    mean %.5f min %.5f (n=%d)" % (pl_arr.mean(), pl_arr.min(), len(pl_arr)))
    print("  rel L2      mean %.5f p90 %.5f" % (rel.mean(), np.percentile(rel, 90)))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
