"""
run_uq_risk.py —— 官方权重 → 参数扰动 MC → 保形校准 → 三级预警，一条命令跑完
==============================================================================

这条管线只回答两个问题，而且都要用真值（`sg_test_u.pt`）反向验证：
  Q1 不确定性有多大、可不可信？
     逐格 band = 未扰动预测 ± κ·std(参数扰动副本)，κ 在校准子集的**羽流内格点**上取
     "满足 PICP≥1−α 的最小值"，再在**没参与选 κ** 的验证子集上报 PICP/MPIW ——
     覆盖率是兑现的还是吹出来的，看这一行。另有 reliability 扫描：整条"承诺 vs 兑现"曲线。
  Q2 分级会不会被模型误差带歪？
     每个风险指标先算保形半径 q，把 [m̂−q, m̂+q] 映到真值队列分位 → 区间判级；
     再和"直接用真值分级"（上帝视角）比一致率。一致率低就说明这套排序在真实误差下不稳。

用法（本机 RTX 3070 实测：整条管线约 **3.6 s/场地**（不含可靠性扫描；含 8 档扫描 ≈5.0 s/场地）；每个场地做 1+8 次前向，按 0.195 s/次估算
GPU 约占一半，另一半是 CPU 上的保形校准与逐场地指标 —— 想提速优先减 `--replicas`，不是减 `--limit`）：
    # 快速验证：100 场地 × 8 副本 ≈ 6 分钟
    .venv/Scripts/python.exe tools/run_uq_risk.py \
        --model_path 数据集/sg_UFNO_4500_trained.pt --data_dir 数据集 \
        --limit 100 --replicas 8 --alpha 0.10
    # 正式数字：--limit 500（整个测试集）≈ 41 分钟（含 8 档可靠性扫描），产物加 _500 后缀
    # --replicas 16 让逐格 std 更稳，但耗时接近翻倍
产物：
    experiments/ufno_uq_report*.json   全部实测数字（含三口径对照 + 各轴区分力体检）
    experiments/ufno_uq_tiers*.csv     逐场地指标 + 分数 + 三种口径的判级
    experiments/ufno_uq*.npz           前 --save-fields 个场地的 mean/std/y/mask/plume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "repro", "ufno"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台别被一个字符打断

import torch  # noqa: E402

from eval_ufno_gas_saturation import _expose_to_main  # noqa: E402
from ufno_reader import open_split  # noqa: E402

from src.risk_model import (RiskReference, RiskLevel, TierPolicy,  # noqa: E402
                            classify_site, extract_indicators, physical_time_days)
from src.uq_model import (PerturbConfig, band_coverage, calibrate_kappa,  # noqa: E402
                          mc_band, pooled_kappa, perturb_inputs, score_intervals)

H, W, T, C = 96, 200, 24, 12


def predict(model, a_np: np.ndarray, device) -> np.ndarray:
    """单个 (H,W,T,C) 输入场 → (H,W,T) 饱和度。ascontiguousarray 是必须的：
    按 stride 还原出来的视图不保证连续，直接喂 torch 会拿到错乱的数据。"""
    x = torch.from_numpy(np.ascontiguousarray(a_np[None])).to(device)
    with torch.no_grad():
        return model(x).view(H, W, T).float().cpu().numpy()


def bt_error(pred_days: float, truth_days: float, horizon: float) -> float:
    """突破时间的非一致性分数：一方从未突破、另一方突破过 = 定性判反，按整个时程计罚。"""
    if np.isnan(pred_days) and np.isnan(truth_days):
        return 0.0
    if np.isnan(pred_days) or np.isnan(truth_days):
        return float(horizon)
    return abs(float(pred_days) - float(truth_days))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def tercile_groups(values: np.ndarray) -> np.ndarray:
    """按羽流规模分三组（Mondrian 用），组边界取自本队列自身分位数。"""
    v = np.asarray(values, float)
    lo, hi = np.percentile(v, [33.33, 66.67])
    return np.digitize(v, [lo, hi]).astype(int)


def fmt_days(ind) -> str:
    """突破时间写进 CSV：从未到达顶行 = 空，别用 nan 混在数字列里。"""
    v = ind.breakthrough_days
    return "" if np.isnan(v) else f"{float(v):.1f}"


def main():
    ap = argparse.ArgumentParser(description="U-FNO 推理 + 参数扰动 UQ + 保形校准 + 三级预警")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=100, help="场地数（0=全部 500）")
    ap.add_argument("--replicas", type=int, default=8, help="参数扰动副本数 K（不含基准那次）")
    ap.add_argument("--alpha", type=float, default=0.10, help="区间不覆盖率，0.10 即 90% 覆盖")
    ap.add_argument("--calib_frac", type=float, default=0.6, help="校准子集占比，其余只用于验证覆盖")
    ap.add_argument("--min_band", type=float, default=0.01, help="band 下限（饱和度绝对值）：不宣称优于它的精度")
    ap.add_argument("--kappa_cap", type=float, default=20.0, help="严格口径 κ 的上限")
    ap.add_argument("--sat_thr", type=float, default=0.01, help="判定含 CO2 的饱和度阈值")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--perm_sigma", type=float, default=0.04, help="对数渗透率扰动 σ")
    ap.add_argument("--red_pct", type=float, default=0.90)
    ap.add_argument("--yellow_pct", type=float, default=0.75)
    ap.add_argument("--out_json", default="experiments/ufno_uq_report.json")
    ap.add_argument("--out_csv", default="experiments/ufno_uq_tiers.csv")
    ap.add_argument("--out_npz", default="experiments/ufno_uq.npz")
    ap.add_argument("--save_fields", type=int, default=10, help="存几个场地的完整场（0=不存）")
    args = ap.parse_args()

    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    pol = TierPolicy(red_pct=args.red_pct, yellow_pct=args.yellow_pct)
    pcfg = PerturbConfig(log_perm_sigma=args.perm_sigma, seed=args.seed)
    rng = np.random.default_rng(pcfg.seed)

    test_a, test_u = open_split(args.data_dir, args.split)
    n = test_a.shape[0] if args.limit <= 0 else min(args.limit, test_a.shape[0])
    _expose_to_main()
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device).eval()
    days = physical_time_days(T)
    horizon = float(days[-1])
    print(f"[uq-risk] device={device} 场地={n} 副本 K={args.replicas} α={args.alpha} "
          f"扰动σ(对数渗透率)={pcfg.log_perm_sigma}")

    centers, stds, truths, masks = [], [], [], []
    ind_pred, ind_truth, rel_single, rel_mean = [], [], [], []
    t0 = time.time()
    for i in range(n):
        a_i = np.array(test_a[i], dtype=np.float32)
        y_i = np.array(test_u[i], dtype=np.float32)
        m2 = a_i[:, :, 0, 0] != 0  # 通道 0（对数渗透率）在 t=0 非零 = 有效储层格点，形状 (H,W)
        base = predict(model, a_i, device)
        reps = np.stack([predict(model, perturb_inputs(a_i, pcfg, rng), device)
                         for _ in range(args.replicas)])
        mc_mean, std = mc_band(reps)
        # 中心一律用**未扰动**的 base：扰动副本的均值会系统性偏离条件正确的预测
        # （实测把它当点预测会让 rel L2 从 0.054 涨到 0.094），扰动只贡献离散度 std。
        masked = np.broadcast_to(m2[:, :, None], (H, W, T))
        y_norm = float(np.linalg.norm(y_i[masked])) + 1e-12
        for pred, bucket in ((base, rel_single), (mc_mean, rel_mean)):
            # 与论文/官方 notebook 同口径：遮罩内 ||pred-y||₂ / ||y||₂，不是 L1
            bucket.append(float(np.linalg.norm(pred[masked] - y_i[masked]) / y_norm))
        centers.append(base); stds.append(std); truths.append(y_i); masks.append(masked)
        ind_pred.append(extract_indicators(base, m2, days=days))
        ind_truth.append(extract_indicators(y_i, m2, days=days))
        if (i + 1) % 10 == 0 or i + 1 == n:
            rate = (time.time() - t0) / (i + 1)
            print(f"  [{i + 1}/{n}] relL2 单次={np.mean(rel_single):.4f} MC均值={np.mean(rel_mean):.4f} "
                  f"({rate:.1f}s/场地, 剩余≈{rate * (n - i - 1) / 60:.1f} 分钟)")

    # ---------------- Q1：band 的覆盖率 ---------------- #
    n_cal = max(5, int(n * args.calib_frac))
    cal, ev = list(range(n_cal)), list(range(n_cal, n))
    mb = args.min_band
    # 同一份 band 有两个口径，必须分开报，否则会得出自相矛盾的结论（本次实测踩到）：
    #   全储层格点 —— 绝大多数是"干格点"（真值≈0、模型也≈0，误差本就很小），
    #                 被它们稀释后，κ 只需 0.19 就能兑现 90% 覆盖；
    #   羽流内格点 —— 风险决策真正看的区域，误差明显更大，κ=0.19 时覆盖只剩 0.37。
    # 所以交付口径用"羽流内"：区间是给含 CO2 的格点兜底的，不是给一片零值区兜底的。
    plume_masks = [masks[j] & (truths[j] > args.sat_thr) for j in range(n)]

    def sel(mk):
        return ([centers[j] for j in cal], [stds[j] for j in cal],
                [truths[j] for j in cal], [mk[j] for j in cal])

    def cov(idxs, k, mk=None):
        if not idxs or k <= 0:
            return float("nan"), float("nan")
        m = masks if mk is None else mk
        c = [band_coverage(centers[j], stds[j], truths[j], m[j], k, min_band=mb) for j in idxs]
        return float(np.mean([x[0] for x in c])), float(np.mean([x[1] for x in c]))

    kappa = pooled_kappa(*sel(masks), args.alpha, min_band=mb)            # 全储层边际口径
    kappa_pl = pooled_kappa(*sel(plume_masks), args.alpha, min_band=mb)    # 羽流内边际口径=交付口径
    k_site, _ = calibrate_kappa(*sel(masks), args.alpha)                   # 严格口径：每场地各自达标
    k_site = min(k_site, args.kappa_cap)

    variants = {}
    for name, k in (("pooled_reservoir", kappa), ("pooled_plume", kappa_pl),
                    ("per_site_strict", k_site)):
        p_all, w_all = cov(ev, k)
        p_pl, w_pl = cov(ev, k, plume_masks)
        variants[name] = {"kappa": k, "picp_reservoir": p_all, "mpiw_reservoir": w_all,
                          "picp_plume": p_pl, "mpiw_plume": w_pl}
    picp_cal = cov(cal, kappa_pl, plume_masks)[0]
    picp_ev, mpiw_ev = cov(ev, kappa_pl, plume_masks)
    p_raw, w_raw = cov(ev, 1.0)
    pp_raw, wp_raw = cov(ev, 1.0, plume_masks)
    plume_share = float(np.mean([pm.sum() / max(int(masks[j].sum()), 1)
                                 for j, pm in enumerate(plume_masks) if j < n]))

    # 可靠性扫描：把"承诺的覆盖率"当横轴，看留出集上的兑现值是否贴着对角线。
    # 每一档都重新在校准子集上选 κ —— 这才是 split conformal 的理论与平作图口径一致的做法。
    reliability = []
    for lvl in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
        k_lvl = pooled_kappa(*sel(plume_masks), 1.0 - lvl, min_band=mb)
        p_l, w_l = cov(ev, k_lvl, plume_masks)
        reliability.append({"nominal": lvl, "kappa": round(float(k_lvl), 4),
                            "picp_eval": round(p_l, 4), "mpiw_eval": round(w_l, 5)})

    # ---------------- Q2：指标级保形半径 + 分级 ---------------- #
    axes = {"breakthrough": ([p.breakthrough_days for p in ind_pred], [t.breakthrough_days for t in ind_truth]),
            "areal": ([p.ratio_peak for p in ind_pred], [t.ratio_peak for t in ind_truth]),
            "lateral": ([p.lateral_share for p in ind_pred], [t.lateral_share for t in ind_truth])}
    q, q_mond, picp_ind = {}, {}, {}
    peak = np.array([p.ratio_peak for p in ind_pred])
    grp = tercile_groups(peak)
    for key, (pv, tv) in axes.items():
        pv, tv = np.array(pv, float), np.array(tv, float)
        if key == "breakthrough":
            sc = np.array([bt_error(a, b, horizon) for a, b in zip(pv, tv)])
        else:
            sc = np.abs(pv - tv)
        r = score_intervals(sc[cal], sc[ev], args.alpha)
        q[key] = r["q"]
        picp_ind[key] = r["picp_eval"]
        rm = score_intervals(sc[cal], sc[ev], args.alpha, cal_groups=grp[cal], eval_groups=grp[ev])
        q_mond[key] = rm["q_per_group"]

    ref_truth = RiskReference(ind_truth)      # 风险轴阈值锚在数值真值上
    ref_pred = RiskReference(ind_pred)        # 可信度轴参照：真值恒 0，必须用预测队列
    sites = []
    for i in range(n):
        c = classify_site(ind_pred[i], ref_truth, pol, q=q, conf_ref=ref_pred)
        c_point = classify_site(ind_pred[i], ref_truth, pol, conf_ref=ref_pred)
        c_true = classify_site(ind_truth[i], ref_truth, pol, conf_ref=ref_pred)
        sites.append({"i": i, "cls": c, "point": c_point, "truth": c_true, "rel_l2": rel_single[i]})

    def count(key):
        lv = [s[key].level for s in sites]
        return {"green": sum(1 for v in lv if v == RiskLevel.GREEN),
                "yellow": sum(1 for v in lv if v == RiskLevel.YELLOW),
                "red": sum(1 for v in lv if v == RiskLevel.RED)}

    agree = sum(1 for s in sites if s["cls"].level == s["truth"].level) / n
    agree_point = sum(1 for s in sites if s["point"].level == s["truth"].level) / n
    rho = spearman([s["cls"].score for s in sites], [s["truth"].score for s in sites])
    # 权重敏感性：换 5 组权重，看排序是否稳（答辩用："综合分不是拍脑袋权重拍出来的"）
    grid = [{"breakthrough": .45, "areal": .35, "lateral": .20},
            {"breakthrough": .34, "areal": .33, "lateral": .33},
            {"breakthrough": .60, "areal": .30, "lateral": .10},
            {"breakthrough": .20, "areal": .50, "lateral": .30},
            {"breakthrough": .50, "areal": .20, "lateral": .30}]
    rho_w = []
    for wts in grid:
        p2 = TierPolicy(weights=dict(wts), red_pct=pol.red_pct, yellow_pct=pol.yellow_pct)
        sp = np.array([sum(p2.weights[k] * ref_truth.axis_percentile(k, getattr(ind_pred[i], f))
                           for k, (f, _) in ref_truth.AXES.items()) for i in range(n)])
        st = np.array([sum(p2.weights[k] * ref_truth.axis_percentile(k, getattr(ind_truth[i], f))
                           for k, (f, _) in ref_truth.AXES.items()) for i in range(n)])
        rho_w.append(spearman(sp, st))

    # 各轴的"区分力体检"：并列太多=这根轴其实分不出人，答辩时得自己先说出来
    def axis_diag(field: str) -> dict:
        v = np.array([getattr(t, field) for t in ind_truth], float)
        fin = v[~np.isnan(v)]
        d = {"nan_share": float(np.isnan(v).mean()), "n_finite": int(fin.size)}
        if fin.size:
            d.update({"min": float(fin.min()), "p50": float(np.median(fin)),
                      "max": float(fin.max()),
                      "tied_at_min_share": float((fin == fin.min()).mean()),
                      "tied_at_max_share": float((fin == fin.max()).mean())})
        return d

    axis_diagnostics = {k: axis_diag(f) for k, (f, _) in RiskReference.AXES.items()}
    axis_diagnostics["composite_score"] = {
        "min": float(ref_truth.score_cohort(pol).min()),
        "p50": float(np.median(ref_truth.score_cohort(pol))),
        "max": float(ref_truth.score_cohort(pol).max())}

    tier_iv, tier_pt, tier_tr = count("cls"), count("point"), count("truth")
    report = {
        "n_sites": n, "replicas": args.replicas, "alpha": args.alpha, "seed": args.seed,
        "device": str(device), "perturb": {"log_perm_sigma": pcfg.log_perm_sigma,
                                           "porosity_sigma": pcfg.porosity_sigma,
                                           "inj_rate_rel": pcfg.inj_rate_rel,
                                           "swi_sigma": pcfg.swi_sigma, "lambda_sigma": pcfg.lambda_sigma,
                                           "corr_len_cells": pcfg.corr_len_cells},
        "rel_l2_point_estimate": float(np.mean(rel_single)),
        "rel_l2_if_mc_mean_used": float(np.mean(rel_mean)),
        "field_band": {"operating_kappa": kappa_pl, "scope": "plume",
                       "min_band": mb, "plume_cell_share_of_reservoir": plume_share,
                       "picp_cal": picp_cal, "picp_eval": picp_ev, "mpiw_eval": mpiw_ev,
                       "variants": variants,
                       "uncalibrated_kappa_1": {"picp_reservoir": p_raw, "mpiw_reservoir": w_raw,
                                                "picp_plume": pp_raw, "mpiw_plume": wp_raw},
                       "n_cal": len(cal), "n_eval": len(ev)},
        "reliability": reliability,
        "indicator_conformal": {"q": q, "picp_eval": picp_ind, "q_mondrian": q_mond},
        "axis_diagnostics": axis_diagnostics,
        "tiers": {"interval": tier_iv, "point": tier_pt, "truth_oracle": tier_tr,
                  # 区间口径把"说不准"全塞进黄档，所以真正有信息量的是"确定档占比"：
                  # (绿+红)/总数 —— 这才是"当前精度能确信归类多少场地"的答案。
                  "decisive_share_interval": (tier_iv["green"] + tier_iv["red"]) / n,
                  "decisive_share_point": (tier_pt["green"] + tier_pt["red"]) / n,
                  "agreement_interval_vs_truth": agree, "agreement_point_vs_truth": agree_point,
                  "spearman_score_pred_vs_truth": rho,
                  "spearman_weight_grid_min": float(np.nanmin(rho_w)),
                  "policy": {"red_pct": pol.red_pct, "yellow_pct": pol.yellow_pct,
                             "low_conf_pct": pol.low_conf_pct, "weights": pol.weights}},
        "indicators_pred": [p.as_dict() for p in ind_pred],
        "indicators_truth": [t.as_dict() for t in ind_truth],
        "seconds_total": time.time() - t0,
    }

    print(f"\n[Q1 逐格 band] 交付口径=羽流内格点（占储层掩膜 {plume_share:.1%}），min_band={mb}")
    print(f"           羽流内 κ={kappa_pl:.2f}：校准 PICP={picp_cal:.3f} → "
          f"验证 PICP={picp_ev:.3f}（目标≥{1 - args.alpha:.2f}） MPIW={mpiw_ev:.4f}")
    print("           口径对照（验证集）：")
    for name, v in variants.items():
        print(f"             {name:<18} κ={v['kappa']:.2f}  "
              f"全储层 PICP={v['picp_reservoir']:.3f}/MPIW={v['mpiw_reservoir']:.4f}  "
              f"羽流内 PICP={v['picp_plume']:.3f}/MPIW={v['mpiw_plume']:.4f}")
    print(f"             κ=1（不校准）      全储层 PICP={p_raw:.3f}  羽流内 PICP={pp_raw:.3f} "
          f"→ 参数扰动 std 在羽流内低估真实误差，需放大到 κ={kappa_pl:.2f}")
    print("[可靠性] 承诺覆盖率 → 留出集兑现（羽流内口径；每档都重新在校准子集上选 κ）")
    print("           " + "  ".join(f"{r['nominal']:.2f}:{r['picp_eval']:.3f}" for r in reliability))
    print(f"[Q2 指标半径] " + "  ".join(f"{k}: q={v:.3f} 覆盖={picp_ind[k]:.2f}" for k, v in q.items()))
    print("[区分力体检]（真值队列）" + "  ".join(
        f"{k}: 并列最优 {v.get('tied_at_min_share', float('nan')):.0%}/并列最差 "
        f"{v.get('tied_at_max_share', float('nan')):.0%}/无突破 {v['nan_share']:.0%}"
        for k, v in axis_diagnostics.items() if k != "composite_score") +
        "  | 综合分区间 "
        f"{axis_diagnostics['composite_score']['min']:.2f}–{axis_diagnostics['composite_score']['max']:.2f}"
        "（阈值取该队列的分位，非加权和绝对值）")
    print(f"[分级] 区间口径 {tier_iv} | 点口径 {tier_pt} | 真值上帝视角 {tier_tr}")
    print(f"       确定档占比（绿+红）：区间口径 {(tier_iv['green'] + tier_iv['red']) / n:.1%} / "
          f"点口径 {(tier_pt['green'] + tier_pt['red']) / n:.1%} —— "
          f"把误差带兑现到 90% 覆盖后，能被确信归类的场地只剩这些")
    print(f"       与真值判级一致率：区间口径 {agree:.1%} / 点口径 {agree_point:.1%}  "
          f"综合分 Spearman={rho:.3f}  权重网格最差 Spearman={np.nanmin(rho_w):.3f}")

    for path in (args.out_json, args.out_csv, args.out_npz):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["site", "rel_l2", "bt_days_pred", "bt_days_truth", "area_pred", "area_truth",
                     "lateral_pred", "outside_pred", "score", "score_lo", "score_hi",
                     "tier_interval", "tier_point", "tier_truth"])
        for s in sites:
            c, p, t = s["cls"], s["point"], s["truth"]
            wr.writerow([s["i"], round(s["rel_l2"], 4),
                         fmt_days(ind_pred[s["i"]]), fmt_days(ind_truth[s["i"]]),
                         round(ind_pred[s["i"]].ratio_peak, 4), round(ind_truth[s["i"]].ratio_peak, 4),
                         round(ind_pred[s["i"]].lateral_share, 4), round(ind_pred[s["i"]].outside_share, 4),
                         round(c.score, 3), round(c.score_lo, 3), round(c.score_hi, 3),
                         c.level.name, p.level.name, t.level.name])
    if args.save_fields > 0:
        k = min(args.save_fields, n)
        # 逐场地的羽流内覆盖（交付口径 κ）：画"覆盖率的分布"而不只是均值，
        # 均值 0.911 可能藏着十几个 0.5 的场地，答辩要能被追问。
        site_cov = [band_coverage(centers[j], stds[j], truths[j], plume_masks[j], kappa_pl, min_band=mb)
                    for j in range(n)]
        np.savez_compressed(args.out_npz,
                            center=np.stack(centers[:k]), std=np.stack(stds[:k]),
                            y=np.stack(truths[:k]), mask=masks[:k],
                            kappa=np.float32(kappa_pl),
                            plume=plume_masks[:k],
                            picp_site=np.array([c[0] for c in site_cov], np.float32),
                            mpiw_site=np.array([c[1] for c in site_cov], np.float32),
                            rel_l2_site=np.array(rel_single, np.float32),
                            score=np.array([s["cls"].score for s in sites], np.float32),
                            score_lo=np.array([s["cls"].score_lo for s in sites], np.float32),
                            score_hi=np.array([s["cls"].score_hi for s in sites], np.float32),
                            level=np.array([int(s["cls"].level) for s in sites], np.int8),
                            level_truth=np.array([int(s["truth"].level) for s in sites], np.int8))
    print(f"[输出] {args.out_json} / {args.out_csv}" +
          (f" / {args.out_npz}（前 {min(args.save_fields, n)} 个场地场数据）" if args.save_fields else ""))


if __name__ == "__main__":
    main()
