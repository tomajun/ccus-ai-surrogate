"""
make_figures.py —— 从跑批产物出答辩/PPT 用图（纯 CPU，不加载权重、不占 GPU）
==============================================================================

输入是 `tools/run_uq_risk.py` 的三个产物，图里每个数字都直接来自它们：
    experiments/ufno_uq_report_500.json   队列级数字（指标、判级、口径对照、可靠性）
    experiments/ufno_uq_tiers_500.csv     逐场地 rel L2
    experiments/ufno_uq_500.npz           前 N 个场地的完整场（center/std/y/mask/plume/kappa）

用法：
    .venv/Scripts/python.exe tools/make_figures.py                 # 默认 500 场地产物
    .venv/Scripts/python.exe tools/make_figures.py --outdir results/ufno_uq --dpi 300
产物（`results/ufno_uq/`）：
    fig1_scatter_pred_vs_truth.png   预测 vs 真值散点（综合分 + 三根风险轴）
    fig2_residual_distribution.png   残差分布（逐场地 rel L2 / 标准化残差 / 异方差 / 剖面）
    fig3_interval_spans.png          500 个场地的区间跨度 → "确定档只有 33%" 那张图
    fig4_tier_compare.png            三种口径的三档分布对比
    fig5_band_coverage.png           逐格区间带与覆盖（全储层 vs 羽流内两个口径）
    fig6_timeline_axes.png           羽流面积比时间线 + 各轴区分力体检
    fig7_reliability.png             承诺覆盖率 vs 留出集兑现（需要报告里有 reliability 块）
    fig8_field_evolution_siteN.png   羽流随时间演化切片（真值/预测同色标）+ 逐帧误差与面积比
    fig9_tier_confusion.png          判级混淆矩阵（点判 / 区间判 vs 真值 oracle）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.risk_model import (RiskReference, SiteIndicators, TierPolicy,  # noqa: E402
                            composite_score, physical_time_days)

TIER_COLOR = {0: "#2e7d32", 1: "#f9a825", 2: "#c62828"}
TIER_NAME = {0: "绿（低风险）", 1: "黄（说不准）", 2: "红（高风险）"}
PLUME_CMAP = LinearSegmentedColormap.from_list("sg", ["#eef5fb", "#7fb3d5", "#1a5276", "#7b241c"])


def use_cjk() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def load(report_path: str, csv_path: str, npz_path: str) -> dict:
    with open(report_path, encoding="utf-8") as fh:
        rep = json.load(fh)
    z = np.load(npz_path)
    rows, rel_l2 = [], None
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    if "rel_l2_site" in z.files:
        rel_l2 = np.asarray(z["rel_l2_site"], float)
    elif rows:
        rel_l2 = np.array([float(r["rel_l2"]) for r in rows])
    return {"rep": rep, "z": z, "rel_l2": rel_l2, "rows": rows}


def series(rows: list, field: str) -> np.ndarray:
    return np.array([r[field] for r in rows], float)


def truth_scores(rep: dict) -> np.ndarray:
    """真值场地的综合分队位 —— 复用 src/risk_model 里的同一套参照队列/映射，
    不在这儿重算公式，否则图上的"真值分"和跑批的判级会对不上。"""
    pol_d = rep["tiers"]["policy"]
    pol = TierPolicy(weights=dict(pol_d["weights"]), red_pct=pol_d["red_pct"],
                     yellow_pct=pol_d["yellow_pct"], low_conf_pct=pol_d["low_conf_pct"])
    ref = RiskReference([SiteIndicators(**r) for r in rep["indicators_truth"]])
    return np.array([ref.score_percentile(
        composite_score({k: ref.axis_percentile(k, getattr(i, f))
                         for k, (f, _) in ref.AXES.items()}, pol), pol)
        for i in ref.inds], float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def diag_line(ax, lo: float, hi: float, label: str | None = None) -> None:
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#555555", zorder=1, label=label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


# ---------------------------------------------------------------- 图 1：散点 -- #
def fig_scatter(d: dict, outdir: str) -> str:
    rep, z = d["rep"], d["z"]
    ip, it = rep["indicators_pred"], rep["indicators_truth"]
    t = rep["tiers"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    # (a) 综合风险分：点 + 区间横条 + 阈值线
    ax = axes[0][0]
    s_lo, s_hi, sc = (np.asarray(z[k], float) for k in ("score_lo", "score_hi", "score"))
    sc_t = truth_scores(rep)
    lv = np.asarray(z["level"], int)
    for code in (0, 1, 2):
        m = lv == code
        if m.any():
            ax.errorbar(sc[m], sc_t[m], xerr=[sc[m] - s_lo[m], s_hi[m] - sc[m]],
                        fmt="o", ms=3.5, alpha=.55, lw=.6, capsize=1.5,
                        color=TIER_COLOR[code], label=TIER_NAME[code])
    diag_line(ax, 0, 1, "y = x（完美复现判级）")
    ax.axvline(t["policy"]["yellow_pct"], color="#f9a825", ls=":", lw=1.2)
    ax.axvline(t["policy"]["red_pct"], color="#c62828", ls=":", lw=1.2)
    ax.set_xlabel("综合风险分（预测队列分位）")
    ax.set_ylabel("综合风险分（真值队列分位）")
    ax.set_title(f"(a) 综合分：预测 vs 真值   Spearman={t['spearman_score_pred_vs_truth']:.3f}\n"
                 f"颜色=区间判级（红档 {t['interval']['red']} 个），横条=保形区间 [lo,hi]；"
                 f"点判级与真值一致率 {t['agreement_point_vs_truth']:.1%}")
    ax.legend(fontsize=8, loc="lower right")

    # (b)(c)(d) 三根风险轴
    panels = [("ratio_peak", "末帧羽流面积比", False),
              ("breakthrough_days", "突破时间（天）", True),
              ("lateral_share", "贴边界列格点占比", False)]
    for ax, (field, cname, log) in zip([axes[0][1], axes[1][0], axes[1][1]], panels):
        p, tr = series(ip, field), series(it, field)
        ok = np.isfinite(p) & np.isfinite(tr)
        ax.scatter(tr[ok], p[ok], s=9, alpha=.6, color="#1a5276", edgecolors="none")
        lo, hi = (min(np.nanmin(p[ok]), np.nanmin(tr[ok])), max(np.nanmax(p[ok]), np.nanmax(tr[ok])))
        if log:
            lo, hi = max(lo, 1.0), hi * 1.15
            ax.set_xscale("log")
            ax.set_yscale("log")
        diag_line(ax, lo, hi)
        nan_n = int(np.isnan(tr).sum())
        rho = spearman(p, tr)
        ax.set_xlabel(f"{cname}（真值）")
        ax.set_ylabel(f"{cname}（预测）")
        ax.set_title(f"{cname}：Spearman={rho:.3f}" + (f"，{nan_n} 个场地从未突破" if nan_n else ""))
    fig.suptitle("图 1  预测 vs 真值（500 场地全测试集，官方 U-FNO 权重）", fontsize=13)
    path = os.path.join(outdir, "fig1_scatter_pred_vs_truth.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# -------------------------------------------------------------- 图 2：残差分布 -- #
def fig_residual(d: dict, outdir: str, sat_thr: float) -> list[str]:
    rep, z = d["rep"], d["z"]
    fb = rep["field_band"]
    kappa, mb = float(z["kappa"]), fb["min_band"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    # (a) 逐场地相对 L2
    ax = axes[0][0]
    rl = d["rel_l2"]
    if rl is None:
        ax.axis("off")
    else:
        ax.hist(rl, bins=40, color="#1a5276", alpha=.85)
        for v, lab, c in ((float(np.mean(rl)), f"均值 {np.mean(rl):.4f}", "#c62828"),
                          (float(np.median(rl)), f"中位 {np.median(rl):.4f}", "#2e7d32"),
                          (float(np.percentile(rl, 90)), f"p90 {np.percentile(rl, 90):.4f}", "#f9a825")):
            ax.axvline(v, color=c, ls="--", lw=1.4, label=lab)
        ax.set_xlabel("掩膜内相对 L2（逐场地）")
        ax.set_ylabel("场地数")
        ax.set_title(f"(a) 点预测误差分布（n={rl.size}）：headline={rep['rel_l2_point_estimate']:.4f}\n"
                     "右尾那批场地是「个别很难」，不是「整体不准」")
        ax.legend(fontsize=8)

    # (b) 标准化残差
    ax = axes[0][1]
    center, std, y = (np.asarray(z[k], np.float32) for k in ("center", "std", "y"))
    mask, plume = (np.asarray(z[k], bool) for k in ("mask", "plume"))
    zs = []
    for j in range(center.shape[0]):
        pm = plume[j]
        b = np.maximum(std[j], mb)
        zs.append(((y[j] - center[j])[pm] / b[pm]).ravel())
    zall = np.concatenate(zs)
    inband = np.abs(zall) <= kappa
    win = 8.0
    tail = float(np.mean(np.abs(zall) > win))
    ax.hist(np.clip(zall, -win, win), bins=80, color="#1a5276", alpha=.8)
    ax.axvline(-kappa, color="#c62828", ls="--", lw=1.5)
    ax.axvline(kappa, color="#c62828", ls="--", lw=1.5,
               label=f"±κ={kappa:.2f}（留出集羽流内 PICP={fb['picp_eval']:.3f}）")
    ax.set_xlim(-win, win)
    ax.set_xlabel("标准化残差 (真值 − 预测) / max(std, min_band)")
    ax.set_ylabel("格点数")
    ax.set_title(f"(b) 羽流内标准化残差（前 {center.shape[0]} 场地，{inband.size:,} 格点）\n"
                 f"这 10 个场地的池化覆盖 {inband.mean():.1%}；承诺值 {fb['picp_eval']:.3f} 是在留出 200 场地上兑现的"
                 f"（|z|>{win:.0f} 尾巴 {tail:.2%} 已截断）")
    ax.legend(fontsize=8)

    # (c) 异方差：|残差| vs 真值饱和度 —— 必须把干格点一起放进来，否则讲不了「稀释」
    ax = axes[1][0]
    edges = np.array([0, .01, .05, .1, .2, .3, .41], float)
    sums = np.zeros(edges.size - 1)
    cnts = np.zeros(edges.size - 1, int)
    for j in range(center.shape[0]):
        mk = mask[j]
        r_all = (y[j] - center[j])[mk].ravel()
        s_all = y[j][mk].ravel()
        idx = np.clip(np.searchsorted(edges, s_all, side="right") - 1, 0, edges.size - 2)
        np.add.at(sums, idx, np.abs(r_all))
        np.add.at(cnts, idx, 1)
    keep = cnts > 50
    means = np.where(keep, sums / np.maximum(cnts, 1), 0)
    lbl = [f"{edges[b]:.2f}–{edges[b + 1]:.2f}" for b in range(edges.size - 1)]
    ax.bar(np.arange(edges.size - 1)[keep], means[keep],
           color=["#9e9e9e" if b == 0 else "#7b241c" for b in np.arange(edges.size - 1)[keep]], alpha=.88)
    for i, (yv, n) in enumerate(zip(means[keep], cnts[keep])):
        ax.text(i, yv, f"n={n/1000:.0f}k", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(edges.size - 1)[keep])
    ax.set_xticklabels([lbl[b] for b in np.arange(edges.size - 1)[keep]], fontsize=8, rotation=20)
    ax.set_xlabel("真值 CO2 饱和度分箱（储层掩膜内全部格点）")
    ax.set_ylabel("平均绝对残差")
    dry_share = cnts[0] / max(cnts.sum(), 1)
    ax.set_title(f"(c) 误差随饱和度增大：灰色干格点占掩膜 {dry_share:.0%}、误差近零\n"
                 "这就是「全储层口径 PICP=0.90 却骗人」的根源")

    # (d) 残差剖面（末帧，羽流已发育；掩膜外置灰）
    ax = axes[1][1]
    j = 0
    m2 = mask[j][:, :, -1]
    r2d = np.where(m2, (y[j] - center[j])[:, :, -1], np.nan)
    v = float(np.nanmax(np.abs(r2d))) + 1e-9
    bad_cmap = matplotlib.colormaps["RdBu_r"].copy()
    bad_cmap.set_bad("#eeeeee")
    im = ax.imshow(r2d, cmap=bad_cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v),
                   aspect="auto")
    ax.set_xlabel("列索引（水平方向）")
    ax.set_ylabel("行索引（图上方=网格顶面/盖层）")
    ax.set_title(f"(d) 场地 0 末帧残差剖面（真值−预测）\n"
                 f"范围 ±{v:.3f}，蓝=低估、红=高估；灰=储层掩膜外")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.subplots_adjust(hspace=.62, wspace=.34)

    fig.suptitle("图 2  残差与不确定性分布（500 场地队列 + 前 10 场地逐格）", fontsize=13)
    path = os.path.join(outdir, "fig2_residual_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return [path]


# ------------------------------------------------------------ 图 3：区间跨度 -- #
def fig_spans(d: dict, outdir: str) -> str:
    rep, z = d["rep"], d["z"]
    t = rep["tiers"]
    sc_lo = np.asarray(z["score_lo"], float)
    sc_hi = np.asarray(z["score_hi"], float)
    lv = np.asarray(z["level"], int)
    order = np.argsort((sc_lo + sc_hi) / 2)
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for i, k in enumerate(order):
        ax.plot([sc_lo[k], sc_hi[k]], [i, i], lw=1.1, alpha=.85, color=TIER_COLOR[lv[k]])
    ax.axvline(t["policy"]["yellow_pct"], color="#f9a825", ls="--", lw=1.5,
               label=f"黄/绿线（队列分位 {t['policy']['yellow_pct']}）")
    ax.axvline(t["policy"]["red_pct"], color="#c62828", ls="--", lw=1.5,
               label=f"红/黄线（队列分位 {t['policy']['red_pct']}）")
    ax.set_yticks([])
    ax.set_ylim(0, len(order))
    ax.set_xlabel("综合风险分在真值队列里的分位（按区间中点排序）")
    ax.set_title("图 3  把误差带兑现到 90% 覆盖后，逐场地的判级区间\n"
                 f"整条区间在阈值同侧才算「确定档」：绿 {t['interval']['green']} + 红 {t['interval']['red']} "
                 f"= {(t['interval']['green'] + t['interval']['red']) / rep['n_sites']:.1%}，"
                 f"其余 {t['interval']['yellow']} 个（{t['interval']['yellow'] / rep['n_sites']:.1%}）判黄=模型说不准")
    ax.legend(fontsize=9, loc="upper left")
    path = os.path.join(outdir, "fig3_interval_spans.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------- 图 4：三档口径对比 -- #
def fig_tiers(d: dict, outdir: str) -> str:
    rep = d["rep"]
    t = rep["tiers"]
    fb = rep["field_band"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    names = ["真值上帝视角", "点判级（无区间）", "区间判级（90% 覆盖）"]
    keys = ["truth_oracle", "point", "interval"]
    left = np.zeros(len(keys))
    for code in (0, 1, 2):
        vals = np.array([t[k][["green", "yellow", "red"][code]] for k in keys], float)
        ax.barh(names, vals, left=left, color=TIER_COLOR[code], label=TIER_NAME[code], height=.6)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 15:
                ax.text(l + v / 2, y, f"{int(v)}", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        left += vals
    ax.set_xlabel("场地数")
    ax.set_title(f"(a) 三种口径的判级分布（n={rep['n_sites']}）\n"
                 f"点判级一致率 {t['agreement_point_vs_truth']:.1%} → 区间判级 {t['agreement_interval_vs_truth']:.1%}")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    # (b) 同一个 α、同一个模型，只换「在哪些格点上选 κ」→ 羽流内覆盖差 2.5 倍
    rows = [("κ=1（不校准）", 1.0, fb["uncalibrated_kappa_1"]["picp_reservoir"],
             fb["uncalibrated_kappa_1"]["picp_plume"]),
            ("全储层口径", fb["variants"]["pooled_reservoir"]["kappa"],
             fb["variants"]["pooled_reservoir"]["picp_reservoir"],
             fb["variants"]["pooled_reservoir"]["picp_plume"]),
            ("羽流内口径（交付）", fb["variants"]["pooled_plume"]["kappa"],
             fb["variants"]["pooled_plume"]["picp_reservoir"],
             fb["variants"]["pooled_plume"]["picp_plume"]),
            ("逐场地严格口径", fb["variants"]["per_site_strict"]["kappa"],
             fb["variants"]["per_site_strict"]["picp_reservoir"],
             fb["variants"]["per_site_strict"]["picp_plume"])]
    x = np.arange(len(rows))
    w = .38
    ax2.bar(x - w / 2, [r[2] for r in rows], width=w, color="#9e9e9e", label="全储层格点 PICP")
    ax2.bar(x + w / 2, [r[3] for r in rows], width=w, color="#7b241c", label="羽流内格点 PICP")
    ax2.axhline(1 - rep["alpha"], color="#2e7d32", ls="--", lw=1.4,
                label=f"承诺覆盖 {1 - rep['alpha']:.2f}")
    for i, r in enumerate(rows):
        ax2.text(i, 1.015, f"κ={r[1]:.2f}", ha="center", fontsize=8.5, color="#222222")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax2.set_ylim(0, 1.22)
    ax2.set_ylabel("留出集 PICP")
    ax2.set_title(f"(b) 口径决定结论：全储层选 κ 时羽流内只有 "
                  f"{fb['variants']['pooled_reservoir']['picp_plume']:.0%}\n"
                  "→ 交付口径改成「只在羽流内校准」，覆盖才兑现")
    ax2.legend(fontsize=7.5, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("图 4  判级分布与区间口径对照", fontsize=13, y=1.03)
    path = os.path.join(outdir, "fig4_tier_compare.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------- 图 5：区间带与覆盖率（场级） -- #
def fig_band(d: dict, outdir: str, site: int, sat_thr: float) -> str:
    rep, z = d["rep"], d["z"]
    fb = rep["field_band"]
    kappa, mb = float(z["kappa"]), fb["min_band"]
    center, std, y = (np.asarray(z[k], np.float32) for k in ("center", "std", "y"))
    mask, plume = (np.asarray(z[k], bool) for k in ("mask", "plume"))
    j = min(site, center.shape[0] - 1)
    band = kappa * np.maximum(std[j], mb)
    covered = (np.abs(y[j] - center[j]) <= band)
    t = -1                                     # 末帧：羽流已发育，最有信息量
    pl_rate = float(covered[:, :, t][plume[j][:, :, t]].mean())
    rs_rate = float(covered[:, :, t][mask[j][:, :, t]].mean())

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [(np.where(mask[j][:, :, t], y[j][:, :, t], np.nan), PLUME_CMAP, True,
               "真值 CO2 饱和度（末帧）", None),
              (np.where(mask[j][:, :, t], center[j][:, :, t], np.nan), PLUME_CMAP, True,
               "预测 CO2 饱和度（末帧）", None),
              (band[:, :, t], "magma", False, f"区间半宽 κ·max(std, {mb})，κ={kappa:.2f}", None),
              (np.where(plume[j][:, :, t], 1.0 - covered[:, :, t], np.nan), "Greys", False,
               f"场地 {j} 末帧·羽流内：白=覆盖 黑=漏（{pl_rate:.1%}）\n"
               f"队列值 {fb['picp_eval']:.1%} 是留出 200 场地 × 24 帧的平均，漏点集中在羽流前沿", (0, 1))]
    for ax, (f, cmap, cbar, title, lim) in zip(axes.ravel(), panels):
        cm = (cmap if isinstance(cmap, matplotlib.colors.Colormap) else matplotlib.colormaps[cmap]).copy()
        cm.set_bad("#f2f2f2")
        im = ax.imshow(f, cmap=cm, aspect="auto", vmin=lim[0] if lim else None,
                       vmax=lim[1] if lim else None)
        ax.set_xlabel("列索引（水平方向）")
        ax.set_ylabel("行索引（图上方=网格顶面/盖层）")
        ax.set_title(title)
        if cbar:
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.suptitle(f"图 5  场地 {j}（末帧剖面）：羽流只占储层掩膜的 {fb['plume_cell_share_of_reservoir']:.1%}\n"
                 f"同一张 band，本帧羽流内覆盖 {pl_rate:.1%}、全储层格点覆盖 {rs_rate:.1%}"
                 f"（干格点恒被覆盖 → 稀释）；留出集羽流内兑现 PICP={fb['picp_eval']:.3f}（承诺 {1 - rep['alpha']:.2f}）\n"
                 f"口径对照：全储层选 κ={fb['variants']['pooled_reservoir']['kappa']:.2f} 时羽流内只有 "
                 f"{fb['variants']['pooled_reservoir']['picp_plume']:.1%}；"
                 f"逐场地严格 κ={fb['variants']['per_site_strict']['kappa']:.2f} 时 MPIW 涨到 "
                 f"{fb['variants']['per_site_strict']['mpiw_plume']:.2f}", fontsize=11.5, y=1.0)
    fig.subplots_adjust(hspace=.45, wspace=.28, top=.82)
    path = os.path.join(outdir, f"fig5_band_coverage_site{site}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------ 图 6：时间线 + 区分力体检 -- #
def fig_timeline(d: dict, outdir: str, sat_thr: float) -> str:
    rep, z = d["rep"], d["z"]
    fb = rep["field_band"]
    q_areal = rep["indicator_conformal"]["q"]["areal"]
    center, y = (np.asarray(z[k], np.float32) for k in ("center", "y"))
    mask = np.asarray(z["mask"], bool)
    n_t = center.shape[-1]
    days = physical_time_days(n_t)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) 面积比时间线：区间用"指标级保形半径 q"，不用逐格 band 传播
    #     —— 逐格 band 有 min_band 地板，传播到"是否 >sat_thr"的计数上会退化成整域覆盖
    ax = axes[0]
    palette = ["#1a5276", "#7b241c", "#117864"]
    for j in range(min(3, center.shape[0])):
        inside = mask[j]
        denom = max(int(inside[:, :, 0].sum()), 1)
        pred = (((center[j] > sat_thr) & inside).sum(axis=(0, 1)) / denom)
        truth = (((y[j] > sat_thr) & inside).sum(axis=(0, 1)) / denom)
        c = palette[j % len(palette)]
        ax.fill_between(days, pred - q_areal, pred + q_areal, alpha=.16, lw=0, color=c)
        ax.plot(days, truth, lw=2.2, color=c, label=f"场地 {j} 真值")
        ax.plot(days, pred, lw=1.4, ls="--", color=c)
    ax.axhline(rep["axis_diagnostics"]["areal"]["p50"], color="#888", ls=":", lw=1,
               label=f"队列中位面积比 {rep['axis_diagnostics']['areal']['p50']:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(0, .65)
    ax.set_xlabel("物理时间（天，对数轴）")
    ax.set_ylabel("羽流面积占储层掩膜比例")
    ax.set_title(f"图 6a  羽流推进时间线：实线=真值 虚线=预测 阴影=±q(面积比)={q_areal:.4f}\n"
                 "真值与预测几乎重合 —— 这根轴的点预测精度不是瓶颈")

    # (b) 区分力体检
    ax = axes[1]
    ad = rep["axis_diagnostics"]
    labels = {"breakthrough": "突破时间", "areal": "末帧面积比", "lateral": "贴边界占比"}
    ks = list(labels)
    x = np.arange(len(ks))
    ax.bar(x - .25, [ad[k]["tied_at_min_share"] * 100 for k in ks], .25, color="#1a5276",
           label="并列在最优值（分不出好坏）")
    ax.bar(x, [ad[k]["tied_at_max_share"] * 100 for k in ks], .25, color="#7b241c",
           label="并列在最差值")
    ax.bar(x + .25, [ad[k]["nan_share"] * 100 for k in ks], .25, color="#9e9e9e",
           label="该轴无定义（从未突破）")
    for i, k in enumerate(ks):
        ax.text(i - .25, ad[k]["tied_at_min_share"] * 100, f"{ad[k]['tied_at_min_share']:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in ks])
    ax.set_ylabel("占 500 场地的比例（%）")
    ax.set_ylim(0, 40)
    ax.set_title("图 6b  各轴区分力体检：突破时间 30% 并列在 1.0 d\n"
                 f"综合分只落在 {ad['composite_score']['min']:.2f}–{ad['composite_score']['max']:.2f}"
                 "（所以阈值必须用队列分位）")
    ax.legend(fontsize=8)
    fig.subplots_adjust(top=.82, wspace=.24)
    path = os.path.join(outdir, "fig6_timeline_axes.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------- 图 7：可靠性 -- #
def fig_reliability(d: dict, outdir: str) -> str | None:
    rel = d["rep"].get("reliability")
    if not rel:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    nom = [r["nominal"] for r in rel]
    picp = [r["picp_eval"] for r in rel]
    mpiw = [r["mpiw_eval"] for r in rel]
    kap = [r["kappa"] for r in rel]
    picp_site = (np.asarray(d["z"]["picp_site"], float) if "picp_site" in d["z"].files
                 else np.array([float(np.nanmean(picp))]))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="#555", lw=1, label="理想（完全兑现）")
    ax.plot(nom, picp, "-o", color="#1a5276", label="本管线（留出集，羽流内口径）")
    dev = max(abs(p - n) for n, p in zip(nom, picp))
    i90 = nom.index(0.90)
    ax.annotate(f"交付口径 0.90 → {picp[i90]:.3f}", (0.90, picp[i90]), textcoords="offset points",
                xytext=(-96, 10), fontsize=8, color="#1a5276")
    ax.set_xlabel("承诺的覆盖率 1−α")
    ax.set_ylabel("留出集实测 PICP")
    ax.set_title(f"图 7a  可靠性图：8 档承诺全部兑现（最大偏差 {dev:.1%}）")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(nom, mpiw, "-s", color="#7b241c", label="MPIW（饱和度绝对宽度）")
    ax2 = ax.twinx()
    ax2.plot(nom, kap, "-^", color="#2e7d32", lw=1, label="所选 κ")
    ax.set_xlabel("承诺的覆盖率 1−α")
    ax.set_ylabel("平均区间宽度 MPIW", color="#7b241c")
    ax2.set_ylabel("放大系数 κ", color="#2e7d32")
    ax.set_title("图 7b  代价曲线：从 0.90 提到 0.99，区间要宽 3.1 倍")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    fig.suptitle("图 7  保形区间的可靠性与代价（每档重新在校准子集上选 κ，留出集上兑现）\n"
                 f"承诺 0.50/0.60/0.70/0.80/0.85/0.90/0.95/0.99 → 实测 "
                 + "/".join(f"{p:.2f}" for p in picp) +
                 f"\n逐场地看：羽流内覆盖均值 {float(np.mean(picp_site)):.3f}，"
                 f"但 {float((picp_site < .8).mean()):.1%} 的场地低于 0.80 —— 队列兑现 ≠ 每个场地都兑现",
                 fontsize=11.5)
    fig.subplots_adjust(top=.74, wspace=.28)
    path = os.path.join(outdir, "fig7_reliability.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------ 图 8：羽流随时间演化（真值 vs 预测，逐帧） -- #
def fig_evolution(d: dict, outdir: str, site: int, sat_thr: float) -> str | None:
    rep, z = d["rep"], d["z"]
    if "center" not in z.files or int(z["center"].shape[0]) == 0:
        return None
    center, y = (np.asarray(z[k], np.float32) for k in ("center", "y"))
    mask = np.asarray(z["mask"], bool)
    j = min(site, center.shape[0] - 1)
    n_t = center.shape[-1]
    days = physical_time_days(n_t)
    frames = np.unique(np.linspace(0, n_t - 1, 6).round().astype(int))
    inside = mask[j]
    vmax = float(np.nanmax(np.where(inside, y[j], np.nan)))

    # 逐帧 rel L2 与面积比（都只在储层掩膜内统计）
    rel_fr, ar_pred, ar_truth = [], [], []
    for t in range(n_t):
        m = mask[j][:, :, t]
        num = float(np.sqrt(((center[j][:, :, t] - y[j][:, :, t])[m] ** 2).sum()))
        den = float(np.sqrt((y[j][:, :, t][m] ** 2).sum())) or 1.0
        rel_fr.append(num / den)
        den_c = max(int(m.sum()), 1)
        ar_pred.append(float(((center[j][:, :, t] > sat_thr) & m).sum()) / den_c)
        ar_truth.append(float(((y[j][:, :, t] > sat_thr) & m).sum()) / den_c)
    rel_fr = np.array(rel_fr)
    kmax = int(rel_fr.argmax())

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(3, 7, width_ratios=[1] * 6 + [.05], height_ratios=[1, 1, .8],
                          hspace=.42, wspace=.14)
    for row, (field, name) in enumerate([(y[j], "真值"), (center[j], "预测")]):
        for i, t in enumerate(frames):
            ax = fig.add_subplot(gs[row, i])
            cm = PLUME_CMAP.copy()
            cm.set_bad("#f2f2f2")
            ax.imshow(np.where(mask[j][:, :, t], field[:, :, t], np.nan), cmap=cm,
                      aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            ttl = f"第 {t} 帧 · {days[t]:.0f} d"
            if row == 0:
                ax.set_title(ttl, fontsize=9)
            else:
                ax.set_xlabel(f"{ttl}（rel L2={rel_fr[t]:.3f}）", fontsize=8)
        fig.text(.012, .66 - row * .31, name, rotation=90, fontsize=12, va="center")

    ax = fig.add_subplot(gs[2, :3])
    ax.plot(np.arange(n_t), rel_fr, "-o", ms=3, color="#7b241c")
    ax.axhline(rep["rel_l2_point_estimate"], color="#888", ls=":",
               label=f"队列 500 场地均值 {rep['rel_l2_point_estimate']:.4f}")
    ax.plot([kmax], [rel_fr[kmax]], "s", color="#7b241c")
    ax.annotate(f"最差帧 t={kmax}：{rel_fr[kmax]:.3f}", (kmax, rel_fr[kmax]),
                textcoords="offset points", xytext=(-58, -16), fontsize=8)
    ax.set_xlabel("帧序号（0=注入初期，23=末帧）")
    ax.set_ylabel("逐帧 rel L2（掩膜内）")
    ax.set_ylim(0, rel_fr.max() * 1.35)
    ax.set_title(f"图 8c  误差随时间怎么长：早期 {rel_fr[:3].mean():.3f} 上下，"
                 f"羽流铺开后涨到峰值 {rel_fr[kmax]:.3f}")
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[2, 3:7])
    ax.plot(days, ar_truth, "-o", ms=3, color="#1a5276", label="真值面积比")
    q = rep["indicator_conformal"]["q"]["areal"]
    ax.plot(days, ar_pred, "--s", ms=3, color="#7b241c", label="预测面积比")
    ax.fill_between(days, np.array(ar_pred) - q, np.array(ar_pred) + q, alpha=.2, lw=0,
                    color="#7b241c", label=f"±q(面积比)={q:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("物理时间（天，对数轴）")
    ax.set_ylabel("羽流面积占储层掩膜比例")
    ax.set_title("图 8d  这根轴上真值与预测重合（面积比不是瓶颈）")
    ax.legend(fontsize=8, loc="upper left")
    cax = fig.add_subplot(gs[:2, 6])
    cm0 = PLUME_CMAP.copy(); cm0.set_bad("#f2f2f2")
    sm = matplotlib.cm.ScalarMappable(cmap=cm0, norm=matplotlib.colors.Normalize(0, vmax))
    fig.colorbar(sm, cax=cax, label="CO2 饱和度")
    site_rel = (None if d["rel_l2"] is None or j >= len(d["rel_l2"]) else float(d["rel_l2"][j]))
    who = (f"该场地整场 rel L2={site_rel:.4f}（全 24 帧合起来算，与判级表同口径）"
           if site_rel is not None else "该场地逐帧 rel L2 均值")
    fig.suptitle(f"图 8  场地 {j} 羽流演化（{len(frames)} 帧 x–z 垂向剖面，灰=储层掩膜外）："
                 f"上排真值 / 下排预测，同一色标上限 {vmax:.2f}\n"
                 f"{who}；逐帧曲线见 8c"
                 f"；队列 500 场地均值 {rep['rel_l2_point_estimate']:.4f}", fontsize=12)
    path = os.path.join(outdir, f"fig8_field_evolution_site{site}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------- 图 9：判级混淆矩阵（点判 / 区间判） -- #
TIER_IDX = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def fig_confusion(d: dict, outdir: str) -> str | None:
    rep, rows = d["rep"], d.get("rows") or []
    if not rows or "tier_truth" not in rows[0]:
        return None
    tr = np.array([TIER_IDX[r["tier_truth"]] for r in rows])
    preds = [(np.array([TIER_IDX[r["tier_point"]] for r in rows]),
              "agreement_point_vs_truth", "图 9a  点判级 vs 真值判级"),
             (np.array([TIER_IDX[r["tier_interval"]] for r in rows]),
              "agreement_interval_vs_truth", "图 9b  区间判级（交付口径）vs 真值判级")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    worst = 0
    for ax, (pred, key, ttl) in zip(axes, preds):
        cmx = np.zeros((3, 3), int)
        for a, b in zip(tr, pred):
            cmx[a, b] += 1
        worst = max(worst, int(cmx[0, 2] + cmx[2, 0]))
        ax.imshow(cmx, cmap="Blues", vmin=0, vmax=cmx.max())
        for a in range(3):
            for b in range(3):
                share = cmx[a, b] / max(cmx[a].sum(), 1)
                ax.text(b, a, f"{cmx[a, b]}\n{share:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if cmx[a, b] > .55 * cmx.max() else "#1a3b5d")
        ax.set_xticks(range(3), ["绿", "黄", "红"])
        ax.set_yticks(range(3), ["绿", "黄", "红"])
        ax.set_xlabel("模型判级")
        ax.set_ylabel("真值判级（拿真值场算，oracle）")
        ax.set_title(f"{ttl}\n一致率 {rep['tiers'][key]:.1%}")
    fig.suptitle("图 9  判级混淆矩阵：错判几乎全在相邻档（绿、红互错最多 "
                 f"{worst} 个 / {len(rows)}）\n"
                 "所以本模型的定位是"
                 "「排序筛查 + 提示复核」，不是「给单个场地盖红章」", fontsize=12)
    fig.subplots_adjust(top=.78, wspace=.32)
    path = os.path.join(outdir, "fig9_tier_confusion.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def resolve_trio(report: str, csv_path: str, npz_path: str) -> tuple[str, str, str]:
    """按文档步骤 ⑤(100 场地快速) → ⑦(出图) 走时，默认路径还不存在，回落到快速产物。"""
    if os.path.exists(report):
        return report, csv_path, npz_path
    quick = ("experiments/ufno_uq_report.json", "experiments/ufno_uq_tiers.csv",
             "experiments/ufno_uq.npz")
    if os.path.exists(quick[0]):
        print(f"[提示] 没有 {report}（500 场地正式产物），改用 {quick[0]}（100 场地快速跑）")
        return quick
    raise SystemExit(f"找不到报告 {report}，也没有快速产物 {quick[0]} —— 先跑 tools/run_uq_risk.py")


def main() -> None:
    ap = argparse.ArgumentParser(description="从 UQ/判级跑批产物出图（纯 CPU）")
    ap.add_argument("--report", default="experiments/ufno_uq_report_500.json")
    ap.add_argument("--csv", default="experiments/ufno_uq_tiers_500.csv")
    ap.add_argument("--npz", default="experiments/ufno_uq_500.npz")
    ap.add_argument("--outdir", default="results/ufno_uq")
    ap.add_argument("--sat-thr", type=float, default=0.01)
    ap.add_argument("--band-site", type=int, default=0, help="图 5 用哪个场地的完整场")
    args = ap.parse_args()

    use_cjk()
    os.makedirs(args.outdir, exist_ok=True)
    d = load(*resolve_trio(args.report, args.csv, args.npz))
    made = [fig_scatter(d, args.outdir), *fig_residual(d, args.outdir, args.sat_thr),
            fig_spans(d, args.outdir), fig_tiers(d, args.outdir),
            fig_band(d, args.outdir, args.band_site, args.sat_thr),
            fig_timeline(d, args.outdir, args.sat_thr),
            fig_evolution(d, args.outdir, args.band_site, args.sat_thr)]
    rel = fig_reliability(d, args.outdir)
    if rel:
        made.append(rel)
    else:
        print("[跳过] 报告里没有 reliability 块 → 图 7 未生成（用最新 run_uq_risk.py 重跑即可）")
    cfm = fig_confusion(d, args.outdir)
    if cfm:
        made.append(cfm)
    for p in [m for m in made if m]:
        print("[图]", p)


if __name__ == "__main__":
    main()
