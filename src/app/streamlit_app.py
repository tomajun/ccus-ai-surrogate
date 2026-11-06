"""
streamlit_app.py —— 演示：官方 U-FNO 权重的复现结果 + 不确定性 + 三级风险判级
==============================================================================

用法（不需要 GPU，也不加载 3300 万参数模型；全部读跑批产物）：
    .venv/Scripts/python.exe -m streamlit run src/app/streamlit_app.py

数据来源（先跑 `tools/run_uq_risk.py`，再跑 `tools/make_figures.py`）：
    experiments/ufno_uq_report_500.json   队列级数字
    experiments/ufno_uq_tiers_500.csv     逐场地指标与判级
    experiments/ufno_uq_500.npz           前 10 个场地的完整场（center/std/y/mask/plume）
    results/ufno_uq/*.png                 成品图（PPT 直接可用）

叙事顺序就是答辩顺序：① 复现可信 → ② 误差有多大且覆盖是兑现的 → ③ 判级能信到什么程度。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.risk_model import physical_time_days  # noqa: E402

TIER_COLOR = {"GREEN": "#2e7d32", "YELLOW": "#f9a825", "RED": "#c62828"}
TIER_CN = {"GREEN": "绿 · 低风险", "YELLOW": "黄 · 模型说不准", "RED": "红 · 高风险"}
PLUME = matplotlib.colors.LinearSegmentedColormap.from_list("sg", ["#eef5fb", "#7fb3d5", "#1a5276", "#7b241c"])
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource(show_spinner=False)
def load_artifacts(report_path: str, csv_path: str, npz_path: str):
    if not os.path.exists(report_path):
        return None
    with open(report_path, encoding="utf-8") as fh:
        rep = json.load(fh)
    df = pd.read_csv(csv_path, encoding="utf-8-sig") if os.path.exists(csv_path) else None
    if os.path.exists(npz_path):
        with np.load(npz_path) as nf:        # 一次性读进内存，避免每次交互重新解压
            z = {k: np.asarray(nf[k]) for k in nf.files}
    else:
        z = None
    return {"rep": rep, "df": df, "z": z}


def field_ax(ax, arr_2d, mask_2d, cmap, title, vmin=None, vmax=None):
    show = np.where(mask_2d, arr_2d, np.nan)
    cm = matplotlib.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    cm.set_bad("#f0f0f0")
    im = ax.imshow(show, cmap=cm, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def page_overview(rep, figdir):
    t, fb = rep["tiers"], rep["field_band"]
    st.title("CO2 封存代理模型：复现、不确定性与三级风险判级")
    st.caption("模型=官方 U-FNO 权重（33,097,829 参数，未自行训练）；"
               "数据=U-FNO 公开测试集 500 个地质实现；网格 (96,200,24)，12 通道输入 → 气相饱和度场。")
    c = st.columns(4)
    c[0].metric("复现指标 rel L2", f"{rep['rel_l2_point_estimate']:.4f}",
                help="掩膜内 ||pred−y||₂ / ||y||₂，500 场地平均；官方权重复现基准")
    c[1].metric("羽流内兑现覆盖 PICP", f"{fb['picp_eval']:.3f}",
                f"承诺 {1 - rep['alpha']:.2f}｜留出 {fb['n_eval']} 场地，未参与选 κ")
    c[2].metric("区间平均宽度 MPIW", f"{fb['mpiw_eval']:.4f}", "饱和度绝对值，越小越有用")
    c[3].metric("确定档占比（区间口径）",
                f"{t['decisive_share_interval']:.0%}", f"点口径 {t['decisive_share_point']:.0%}")
    st.markdown(
        f"> **一句话结论**：这个代理模型**足以做封存风险的筛查排序**（综合分 Spearman "
        f"{t['spearman_score_pred_vs_truth']:.3f}，换权重最差 {t['spearman_weight_grid_min']:.3f}），"
        f"**不足以对单个场地下红/绿结论**（把覆盖兑现到 90% 后，"
        f"{t['interval']['yellow']} 个场地的区间跨过阈值线，红档 {t['interval']['red']} 个）。")
    st.markdown("#### 图 1 · 预测 vs 真值")
    st.image(os.path.join(figdir, "fig1_scatter_pred_vs_truth.png"))
    st.markdown("#### 图 8 · 羽流随时间演化（真值 vs 预测，同一色标）")
    f8 = sorted(f for f in os.listdir(figdir) if f.startswith("fig8_field_evolution")) if os.path.isdir(figdir) else []
    if f8:
        st.image(os.path.join(figdir, f8[0]))
    else:
        st.info("缺 fig8 —— 跑一次 `python tools/make_figures.py` 即可生成。")
    st.markdown("#### 图 2 · 残差与不确定性分布")
    st.image(os.path.join(figdir, "fig2_residual_distribution.png"))
    with st.expander("这些数字怎么来的（可展开核对）"):
        st.markdown(
            "- **rel L2**：`repro/ufno/eval_ufno_gas_saturation.py` 与本管线独立算出同一个 "
            f"{rep['rel_l2_point_estimate']:.4f} → 管线自校验。\n"
            f"- **PICP/MPIW**：band = 未扰动预测 ± κ·std(8 组参数扰动副本)，κ 在 {fb['n_cal']} 个校准场地的"
            f"**羽流内格点**上取满足 PICP≥{1 - rep['alpha']:.2f} 的最小值（当前 κ={fb['operating_kappa']:.2f}），"
            f"再在另外 {fb['n_eval']} 个场地上报覆盖 —— 覆盖率是兑现的，不是自报的。\n"
            f"- **为什么不用深度集成**：官方只放 1 组权重，K 个成员必须自训，而训练集 99.5 GB 本机放不下。"
            f"所以不确定性来自**输入侧地质参数扰动**（渗透率/孔隙度/注气速率/Swi/流度比，"
            f"空间相关噪声，相关长度 {rep['perturb']['corr_len_cells']} 格）+ **保形校准**补模型侧误差。")


def page_uq(rep, z, figdir):
    fb = rep["field_band"]
    st.header("不确定性：覆盖率必须连着「口径」一起报")
    st.markdown(
        f"储层掩膜内只有 **{fb['plume_cell_share_of_reservoir']:.1%}** 的格点真含 CO2"
        f"（阈值 {fb['min_band']}），其余是「真值≈0、模型也≈0」的干格点。"
        "拿全部格点选 κ 会得到很漂亮的假覆盖率，羽流里其实没兜住。")
    rows = []
    for name, cn in (("pooled_reservoir", "全储层格点选 κ（错）"),
                     ("pooled_plume", "羽流内格点选 κ（交付口径）"),
                     ("per_site_strict", "逐场地严格达标（过保守）")):
        v = fb["variants"][name]
        rows.append({"口径": cn, "κ": round(v["kappa"], 3),
                     "全储层 PICP": round(v["picp_reservoir"], 3),
                     "全储层 MPIW": round(v["mpiw_reservoir"], 4),
                     "羽流内 PICP": round(v["picp_plume"], 3),
                     "羽流内 MPIW": round(v["mpiw_plume"], 4)})
    u = fb["uncalibrated_kappa_1"]
    rows.insert(0, {"口径": "κ=1（不做保形校准）", "κ": 1.0,
                    "全储层 PICP": round(u["picp_reservoir"], 3),
                    "全储层 MPIW": round(u["mpiw_reservoir"], 4),
                    "羽流内 PICP": round(u["picp_plume"], 3),
                    "羽流内 MPIW": round(u["mpiw_plume"], 4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.image(os.path.join(figdir, "fig4_tier_compare.png"))
    rel = rep.get("reliability")
    if rel:
        st.markdown("#### 可靠性：承诺 vs 兑现（每档重新在校准子集上选 κ）")
        st.dataframe(pd.DataFrame(rel), hide_index=True, width="stretch")
        if os.path.exists(os.path.join(figdir, "fig7_reliability.png")):
            st.image(os.path.join(figdir, "fig7_reliability.png"))
    else:
        st.info("报告里还没有 reliability 块 —— 用最新版 `tools/run_uq_risk.py` 重跑一次即可生成图 7。")
    st.markdown("#### 逐格区间带与覆盖（场地级）")
    st.image(os.path.join(figdir, "fig5_band_coverage_site0.png"))
    if z is not None and "picp_site" in z:
        st.markdown("#### 逐场地覆盖分布（均值 0.91 会不会藏住个别差的场地？）")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(np.asarray(z["picp_site"], float), bins=40, color="#1a5276")
        ax.axvline(fb["picp_eval"], color="#c62828", ls="--", label=f"队列均值 {fb['picp_eval']:.3f}")
        ax.set_xlabel("该场地羽流内 PICP（κ=交付值）")
        ax.set_ylabel("场地数")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


def page_site(rep, z, df):
    st.header("场地浏览器：一个地质实现的预测、区间与判级")
    if z is None:
        st.error("缺少 npz 场数据产物，请先跑 `tools/run_uq_risk.py --save_fields 10`。")
        return
    center, std, y = (np.asarray(z[k], np.float32) for k in ("center", "std", "y"))
    mask, plume = (np.asarray(z[k], bool) for k in ("mask", "plume"))
    kappa, mb = float(z["kappa"]), rep["field_band"]["min_band"]
    n_show = center.shape[0]
    j = st.selectbox("场地编号（npz 里存了前几个就能选几个）", list(range(n_show)),
                     format_func=lambda i: f"场地 {i}（判级 {df.iloc[i]['tier_interval']}）"
                     if df is not None else f"场地 {i}")
    t = st.slider("输出时刻（共 %d 帧）" % center.shape[-1], 0, center.shape[-1] - 1, center.shape[-1] - 1)
    band = kappa * np.maximum(std[j], mb)
    cov = (np.abs(y[j] - center[j]) <= band)[:, :, t]
    m2 = mask[j][:, :, t]
    vmax = float(max(center[j][:, :, t].max(), y[j][:, :, t].max()))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    field_ax(axes[0][0], y[j][:, :, t], m2, PLUME, "真值 CO2 饱和度", 0, vmax)
    field_ax(axes[0][1], center[j][:, :, t], m2, PLUME, "预测 CO2 饱和度", 0, vmax)
    field_ax(axes[1][0], (y[j] - center[j])[:, :, t], m2, "RdBu_r", "残差（真值−预测）", -vmax / 4, vmax / 4)
    field_ax(axes[1][1], np.where(plume[j][:, :, t], 1.0 - cov, np.nan), m2, "Greys",
             "羽流内格点：白=区间覆盖 黑=漏", 0, 1)
    fig.subplots_adjust(hspace=.4, wspace=.25)
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    if df is not None:
        r = df.iloc[j]
        c1.markdown("**判级与分数**")
        c1.dataframe(pd.DataFrame([
            {"口径": "区间判级（交付）", "档位": r["tier_interval"], "综合分": f"{r['score']:.3f}",
             "区间": f"[{r['score_lo']:.3f}, {r['score_hi']:.3f}]"},
            {"口径": "点判级（无区间）", "档位": r["tier_point"], "综合分": f"{r['score']:.3f}", "区间": "—"},
            {"口径": "真值上帝视角", "档位": r["tier_truth"], "综合分": "—", "区间": "—"},
        ]), hide_index=True, width="stretch")
        c2.markdown("**逐场地指标（预测 vs 真值）**")
        ip, it = rep["indicators_pred"][j], rep["indicators_truth"][j]
        c2.dataframe(pd.DataFrame([{"指标": k, "预测": round(v, 4), "真值": round(it[k], 4)}
                                   for k, v in ip.items()]), hide_index=True, width="stretch")
    q = rep["indicator_conformal"]["q"]
    days = physical_time_days(center.shape[-1])
    inside = mask[j]
    denom = max(int(inside[:, :, 0].sum()), 1)
    pred = (((center[j] > mb) & inside).sum(axis=(0, 1)) / denom)
    truth = (((y[j] > mb) & inside).sum(axis=(0, 1)) / denom)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(days, pred - q["areal"], pred + q["areal"], alpha=.2, color="#1a5276")
    ax.plot(days, truth, lw=2, label="真值面积比")
    ax.plot(days, pred, lw=1.4, ls="--", label="预测面积比")
    ax.set_xscale("log")
    ax.set_xlabel("物理时间（天）")
    ax.set_ylabel("羽流占储层面积比")
    ax.set_title(f"场地 {j} 羽流推进（阴影=±q(面积比)={q['areal']:.4f}，90% 覆盖）")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def page_tiers(rep, df, figdir):
    st.header("三级判级：排序可信，单场结论不可信")
    t = rep["tiers"]
    st.markdown(
        f"三根风险轴（权重自动归一）：突破时间 {t['policy']['weights']['breakthrough']} / "
        f"末帧面积比 {t['policy']['weights']['areal']} / 贴边界占比 {t['policy']['weights']['lateral']}。"
        f"每根轴先换成**真值队列分位**再加权，红=队列前 {1 - t['policy']['red_pct']:.0%}、"
        f"黄=前 {1 - t['policy']['yellow_pct']:.0%}；再用保形半径把点判级变区间判级 —— "
        "**整条区间在阈值同侧才给确定档，跨线一律判黄**。")
    st.image(os.path.join(figdir, "fig3_interval_spans.png"))
    st.image(os.path.join(figdir, "fig9_tier_confusion.png"))
    st.image(os.path.join(figdir, "fig6_timeline_axes.png"))
    if df is not None:
        st.markdown("#### 500 个场地的判级明细（可筛选/排序/下载）")
        pick = st.multiselect("只看这些档位", ["GREEN", "YELLOW", "RED"], default=["RED", "GREEN"])
        view = df[df["tier_interval"].isin(pick)] if pick else df
        st.dataframe(view, hide_index=True, width="stretch", height=340)
        st.download_button("下载 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "ufno_uq_tiers_500.csv", "text/csv")


def page_limits(rep):
    t, fb = rep["tiers"], rep["field_band"]
    st.header("边界与红线（答辩时主动说，别被问出来）")
    st.markdown(f"""
| 事项 | 现状 | 说法 |
|---|---|---|
| 模型是否自训 | 否，官方权重推理 | 训练集 99.5 GB 本机放不下；**别说"我们训练了 U-FNO"** |
| 不确定性来源 | 输入侧参数扰动 + 保形校准 | 这是**地质非均质性 + 模型误差的近似**，不是权重级不确定性（深度集成做不了） |
| 三级预警的含义 | 封存风险**筛查排序** | 数据集无井位/断层几何/应力 → **不能说"泄漏预警"** |
| 红档为什么是 0 | 区间口径下 {t['interval']['red']} 个 | 指标保形半径比档位间隔宽；这是 UQ 正常起作用，不是方法失效 |
| 绝对几何阈值 | 已废弃 | 实测 69% 的帧"前锋距顶 0 行"，储层就在网格顶部 → 没有区分力 |
| 真实场地迁移 | 未做 | 全部结论建立在公开数值解分布上（Sleipner 迁移是后续工作） |
| 精度地板 | min_band={fb['min_band']} | 低于它的宽度我们不宣称有用，所以 band 有下限 |

**如果只记一句**：*我们证明了当前代理模型的精度够做筛查排序，不够对单个场地下红/绿结论 —— 这就是不确定性量化给出的答案。*
""")


def main():
    st.set_page_config(page_title="CCUS 封存代理模型 · UQ 与三级风险", layout="wide")
    with st.sidebar:
        st.subheader("数据源")
        report_path = st.text_input("报告 JSON", "experiments/ufno_uq_report_500.json")
        csv_path = st.text_input("逐场地 CSV", "experiments/ufno_uq_tiers_500.csv")
        npz_path = st.text_input("场数据 NPZ", "experiments/ufno_uq_500.npz")
        figdir = st.text_input("成品图目录", "results/ufno_uq")
        st.caption("路径相对仓库根目录。缺哪个就先跑对应脚本。")
    data = load_artifacts(os.path.join(ROOT, report_path) if not os.path.isabs(report_path) else report_path,
                          os.path.join(ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path,
                          os.path.join(ROOT, npz_path) if not os.path.isabs(npz_path) else npz_path)
    if data is None:
        st.error("找不到跑批报告。先执行：\n\n"
                 "`python tools/run_uq_risk.py --model_path 数据集/sg_UFNO_4500_trained.pt "
                 "--data_dir 数据集 --limit 500 --replicas 8 --alpha 0.10`\n\n"
                 "再执行 `python tools/make_figures.py` 出图。")
        return
    fig_abs = os.path.join(ROOT, figdir) if not os.path.isabs(figdir) else figdir
    rep, z, df = data["rep"], data["z"], data["df"]
    page = st.radio("页面", ["① 总览", "② 不确定性", "③ 场地浏览器", "④ 三级判级", "⑤ 边界与红线"],
                    horizontal=True)
    if page.startswith("①"):
        page_overview(rep, fig_abs)
    elif page.startswith("②"):
        page_uq(rep, z, fig_abs)
    elif page.startswith("③"):
        page_site(rep, z, df)
    elif page.startswith("④"):
        page_tiers(rep, df, fig_abs)
    else:
        page_limits(rep)


if __name__ == "__main__":
    main()
