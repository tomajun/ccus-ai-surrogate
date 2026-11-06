"""
risk_model_demo.py —— 把 U-FNO 真实输出接进三级预警，并从真实羽流统计反推阈值
==============================================================================

为什么需要它：`src/risk_model.py` 的阈值（caprock_frac / sat_threshold 等）当初是人为
设定的占位值，从没对照真实 CO2 饱和度场看过一眼。本脚本做三件事：
  1) `--mode timeline` ：对给定样本逐时间步（24 步）跑旧的 assess()，输出绿/黄/红时间线；
  2) `--mode calibrate`：在一批样本上统计羽流面积比、最大饱和度、前锋距储层顶界距离等
     分布的分位数，用来判断"绝对几何阈值"到底有没有区分力（实测结论：没有，见下）；
  3) `--mode tier`    ：新口径 —— 跨样本相对分位 + 可信度降级，逐个场地打印判级理由，
     并与"用真值分级"的上帝视角对照。带保形区间的完整版本在 tools/run_uq_risk.py。

数据来源（二选一）：
  A. 官方预训练权重推理产物（真正的复现结果）：
     python repro/ufno/eval_ufno_gas_saturation.py --model_path <权重> --data_dir 数据集 \
            --limit 20 --save_npz experiments/ufno_pred.npz
     python src/risk_model_demo.py --npz experiments/ufno_pred.npz --pt_key pred --mode calibrate
  B. 直接读官方 .pt 的真值场（不等权重就能先把几何与阈值这套流程调通）：
     python src/risk_model_demo.py --from_pt 数据集 --mode calibrate --limit 20

实测几何（2026-10-29 在本机数据集上验证，与官方 eval notebook 一致）：
  输入 a = (N, 96, 200, 24, 12)，输出 u = (N, 96, 200, 24)
  轴序 = (竖向 96, 水平 200, 时间 24)；竖向 0..200 m，水平 0..3500 m
  掩膜 mask = (a[:, :, :, 0, 0] != 0)，为 0 处在储层之外（盖层/围岩侧）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.risk_model import (RiskConfig, RiskReference, SiteIndicators, TierPolicy,  # noqa: E402
                            assess, classify_site, extract_indicators)  # noqa: E402

H, W, T, C = 96, 200, 24, 12
D_Z_MAX_M, D_X_MAX_M = 200.0, 3500.0
M_PER_ROW = D_Z_MAX_M / H


def physical_time_days() -> np.ndarray:
    """24 个输出时刻对应的注入天数（官方 notebook 同款公式）。"""
    return np.cumsum(np.power(1.421245, range(T)))


def crop_to_reservoir(field_2d: np.ndarray, mask_2d: np.ndarray):
    """把羽流场裁到储层包围盒。

    三级预警里 y=0 必须是"储层顶"而不是整个网格顶，否则盖层带的物理含义就错了：
    原始网格上方还有一堆围岩行，不归一化就会把'离盖层很近'误判成'离网格顶很远'。
    """
    rows = np.where(mask_2d.any(axis=1))[0]
    cols = np.where(mask_2d.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None, 0, 0
    return (field_2d[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1],
            int(rows[0]), int(rows[-1]))


def caprock_breach_fraction(sg_3d: np.ndarray, mask_2d: np.ndarray, sat_thr: float) -> float:
    """羽流出现在储层掩膜之外的占比 —— 比"猜一个盖层带"更硬的信号。

    ⚠️ 模型在掩膜外没有训练监督，那里的非零值可能只是预测噪声，所以它是需要
    人工复核的**指示量**，不能直接当"已泄漏"的证据。
    """
    over = float((np.abs(sg_3d) > sat_thr).sum())
    if over == 0:
        return 0.0
    outside = ~mask_2d[:, :, None]
    return float(((np.abs(sg_3d) > sat_thr) & outside).sum() / over)


def front_gap(field_2d: np.ndarray, mask_2d: np.ndarray, sat_thr: float):
    """前锋（最靠顶的含 CO2 行）到储层顶界的距离：(行数, 米)，无羽流返回 None。"""
    active = np.where(mask_2d.any(axis=1))[0]
    if active.size == 0:
        return None
    hit = np.where(((field_2d > sat_thr) & mask_2d).any(axis=1))[0]
    if hit.size == 0:
        return None
    rows = int(hit[0] - active[0])
    return rows, float(rows * M_PER_ROW)


def run_timeline(sg: np.ndarray, mask_2d: np.ndarray, sat_thr: float, top_frac: float) -> None:
    times = physical_time_days()
    nrows = int(mask_2d[:, 0].sum())
    print(f"储层厚度: {nrows}/{H} 行 ≈ {nrows * M_PER_ROW:.0f} m   "
          f"水平跨度 ≈ {D_X_MAX_M:.0f} m   盖层带阈值 top_frac={top_frac}")
    print(f"{'时间':>9} {'级别':<14} {'机制':<8} 主因")
    for t in range(T):
        crop, _, _ = crop_to_reservoir(sg[:, :, t], mask_2d)
        if crop is None:
            print(f"{times[t]:>7.0f}d  (无有效储层)")
            continue
        r = assess(crop, config=RiskConfig(caprock_frac=top_frac, sat_threshold=sat_thr))
        mechs = ",".join(str(m) for m in r.mechanism_ids) or "-"
        reason = r.reasons[0].split("，")[0] if r.reasons else "-"
        print(f"{times[t]:>7.0f}d  {r.level.label:<14} {mechs:<8} {reason}")


def run_calibrate(cases: list, sat_thr: float) -> None:
    frac, mx, gap_rows, gap_m, cap_sig = [], [], [], [], []
    for sg, mask_2d in cases:
        for t in range(T):
            f2 = sg[:, :, t]
            denom = max(int(mask_2d.sum()), 1)
            frac.append(float(((f2 > sat_thr) & mask_2d).sum() / denom))
            mx.append(float(f2[mask_2d].max()) if mask_2d.any() else 0.0)
            gap = front_gap(f2, mask_2d, sat_thr)
            if gap:
                gap_rows.append(gap[0])
                gap_m.append(gap[1])
        cap_sig.append(caprock_breach_fraction(sg, mask_2d, sat_thr))

    def pct(v, name, unit=""):
        if not v:
            print(f"  {name}: 无样本")
            return
        arr = np.asarray(v, dtype=np.float64)
        print(f"  {name:<22} p50={np.percentile(arr, 50):.4f} p75={np.percentile(arr, 75):.4f} "
              f"p90={np.percentile(arr, 90):.4f} p99={np.percentile(arr, 99):.4f} "
              f"max={arr.max():.4f}{unit}")

    print(f"[calibrate] 口径：{len(cases)} 个样本 × {T} 个时间步")
    pct(frac, "羽流占储层面积比")
    pct(mx, "储层内最大饱和度")
    pct(gap_rows, "前锋距顶(行)")
    pct(gap_m, "前锋距顶(m)", " m")
    pct(cap_sig, "掩膜外信号占比")

    print("\n阈值建议（把分位数填进 RiskConfig / 命令行）：")
    frac_arr = np.asarray(frac, dtype=np.float64)
    cap_arr = np.asarray(cap_sig, dtype=np.float64)
    verdict_lines = []
    if gap_rows:
        g = np.asarray(gap_rows, dtype=np.float64)
        zero_share = float((g == 0).mean())
        case_min = [min([(front_gap(sg[:, :, t], m, sat_thr) or (H,))[0] for t in range(T)])
                    for sg, m in cases]
        reach_share = float(np.mean([v == 0 for v in case_min]))
        print(f"  前锋距顶：0 行帧占比 {zero_share:.1%}；至少有一帧顶到上界的样本占比 {reach_share:.1%}")
        print(f"  前锋距顶分位：p10={np.percentile(g,10):.0f} p25={np.percentile(g,25):.0f} "
              f"p50={np.percentile(g,50):.0f} p75={np.percentile(g,75):.0f} p90={np.percentile(g,90):.0f} 行")
        if zero_share >= 0.25:
            print(f"  --top_frac ≈ {float(np.percentile(g, 10)) / H:.3f}   # 前锋距顶 p10（注意：p10=0 时该阈值等于'顶到就红'）")
            verdict_lines.append(
                "绝对几何带（前锋距顶）在本数据集上区分力不足："
                f"{zero_share:.0%} 的帧 gap=0、{reach_share:.0%} 的样本至少一帧顶到上界，"
                "用固定比例带会把大量样本直接判红。")
        else:
            verdict_lines.append(
                f"绝对几何带有区分力：仅 {zero_share:.0%} 的帧 gap=0，"
                f"{reach_share:.0%} 的样本顶到上界，可按分位数设红/黄带。")
    if frac:
        print(f"  sat_threshold 当前 {sat_thr}；羽流面积比 p50={np.percentile(frac_arr,50):.4f}、"
              f"p90={np.percentile(frac_arr,90):.4f}，想只对'明显羽流'报警就抬到 p75~p90")
    print(
        "\n[实测口径说明]（本段文字由本次运行现场计算，不是预设结论）\n"
        f"  · 羽流面积比：p50={np.percentile(frac_arr,50):.4f} / p90={np.percentile(frac_arr,90):.4f} / "
        f"max={frac_arr.max():.4f} —— 跨样本确有数量级差异，适合做筛查排序。\n"
        f"  · 掩膜外信号占比：p50={np.percentile(cap_arr,50):.4f} / max={cap_arr.max():.4f} —— "
        "官方 notebook 的 mask=(ch0!=0) 只是'有效储层行裁剪'，"
        "既不是储层/盖层分界，也不含废弃井或断层几何；数据里没有井位/断层，机制 2、3 无法取证。\n"
        "  · 因此：" + " ".join(verdict_lines or ["以上判据均具备区分力。"]) + "\n"
        "  · 答辩口径：不要说'盖层失效预警'，要说'按羽流规模与突破时间做封存风险筛查排序'；"
        "若要做真正的盖层/井筒风险评估，需要另找含井与断层几何的场地模型。"
    )


def composite_of(ind: SiteIndicators, ref: RiskReference, policy: TierPolicy) -> float:
    return sum(policy.weights[k] * ref.axis_percentile(k, getattr(ind, f))
               for k, (f, _) in ref.AXES.items())


def run_tier(pred_cases: list, truth_cases: list, sat_thr: float, sample: int,
             policy: TierPolicy) -> None:
    """新口径的逐场地判级（答辩演示用）。

    与 run_timeline 的区别要说清：timeline 走旧的**绝对几何带** assess()，本函数走
    **跨样本相对分位 + 可信度降级** —— 实测有区分力的是后者。
    这里不套保形区间（q=None）：区间要整批样本跑推理+参数扰动才能算，那是
    tools/run_uq_risk.py 的活；本函数的价值是把"为什么判这一档"的理由逐条打出来。
    """
    ind_pred = [extract_indicators(sg, m, sat_thr=sat_thr) for sg, m in pred_cases]
    ind_truth = [extract_indicators(sg, m, sat_thr=sat_thr) for sg, m in truth_cases]
    ref_truth = RiskReference(ind_truth)    # 风险轴锚在真值队列
    ref_pred = RiskReference(ind_pred)      # 可信度轴必须锚在预测队列（真值恒 0 是退化分布）
    n = len(ind_pred)
    print(f"[tier] 参照队列 = {n} 个场地的**真值**指标；红线 = 综合分的队列前 "
          f"{1 - policy.red_pct:.0%}，黄线 = 前 {1 - policy.yellow_pct:.0%}")
    print("       这里只有点判级；带保形区间的判级与覆盖率在 tools/run_uq_risk.py 的产物里\n")

    def pair(i):
        return (classify_site(ind_pred[i], ref_truth, policy, conf_ref=ref_pred),
                classify_site(ind_truth[i], ref_truth, policy, conf_ref=ref_pred))

    order = np.argsort([-composite_of(p, ref_truth, policy) for p in ind_pred])
    agree = 0
    for rank, i in enumerate(order, 1):
        c, t = pair(int(i))
        agree += int(c.level == t.level)
        if rank <= 10 or int(i) == sample:
            mark = ">>" if int(i) == sample else "  "
            print(f"{mark} 排名 {rank:>3} 场地 {int(i):>3}  "
                  f"{'一致' if c.level == t.level else '不一致'}\n    [预测]{c.level.label} "
                  f"{c.score:.3f} | [真值]{t.level.label} {t.score:.3f}\n    " +
                  "\n    ".join(c.reasons) + "\n")
    print(f"[一致率] 预测判级 vs 真值判级 {agree}/{n} = {agree / n:.1%}"
          "（点判级口径；区间口径会把这个数字压低，因为'说不准'一律归黄）")


def cases_from_npz(path: str, key: str, limit: int):
    d = np.load(path)
    n = min(limit, d[key].shape[0]) if limit > 0 else d[key].shape[0]
    out = []
    for i in range(n):
        sg = d[key][i]
        mask_2d = np.asarray(d["mask"][i][:, :, 0] > 0)
        out.append((np.asarray(sg, np.float32), mask_2d))
    return out, n


def cases_from_pt(data_dir: str, split: str, limit: int):
    sys.path.insert(0, os.path.join(ROOT, "repro", "ufno"))
    from ufno_reader import open_split  # 按偏移分页直读，绕开 torch mmap 段错误
    a, u = open_split(data_dir, split)
    n = min(limit, a.shape[0]) if limit > 0 else a.shape[0]
    out = []
    for i in range(n):
        ai = a[i]
        mask_2d = ai[:, :, 0, 0] != 0
        out.append((np.array(u[i], dtype=np.float32), mask_2d))
    return out, n


def main():
    ap = argparse.ArgumentParser(description="U-FNO 输出 -> 三级预警：时间线与阈值校准")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="eval 脚本 --save_npz 的产物")
    src.add_argument("--from_pt", help="官方 .pt 目录（用真值场，不等权重即可调通流程）")
    ap.add_argument("--pt_key", default="pred", choices=["pred", "y"], help="配合 --npz 用预测场还是真值场")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--mode", default="timeline", choices=["timeline", "calibrate", "tier"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sat_thr", type=float, default=0.01)
    ap.add_argument("--top_frac", type=float, default=0.10)
    ap.add_argument("--red_pct", type=float, default=0.90)
    ap.add_argument("--yellow_pct", type=float, default=0.75)
    args = ap.parse_args()

    if args.mode == "timeline":
        if args.npz:
            cases, _ = cases_from_npz(args.npz, args.pt_key, args.sample + 1)
        else:
            cases, _ = cases_from_pt(args.from_pt, args.split, args.sample + 1)
        sg, mask_2d = cases[min(args.sample, len(cases) - 1)]
        run_timeline(sg, mask_2d, args.sat_thr, args.top_frac)
    elif args.mode == "tier":
        if not args.npz:
            ap.error("--mode tier 需要 --npz：要同一批场地的预测场与真值场才能对照")
        pred, n = cases_from_npz(args.npz, "pred", args.limit)
        truth, _ = cases_from_npz(args.npz, "y", args.limit)
        run_tier(pred, truth, args.sat_thr, args.sample,
                 TierPolicy(red_pct=args.red_pct, yellow_pct=args.yellow_pct))
    else:
        cases, n = (cases_from_npz(args.npz, args.pt_key, args.limit) if args.npz
                    else cases_from_pt(args.from_pt, args.split, args.limit))
        print(f"[calibrate] 使用 {n} 个样本（{'预测场' if args.npz else '真值场'}）")
        run_calibrate(cases, args.sat_thr)


if __name__ == "__main__":
    main()
