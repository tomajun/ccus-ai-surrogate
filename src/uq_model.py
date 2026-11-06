"""
uq_model.py —— 单权重代理模型的不确定性量化（UQ）
=================================================================

为什么不是深度集成（诚实前提，答辩会被问）：
    官方只发布了 1 组权重（`sg_UFNO_4500_trained`），K 个成员需要 K 次训练，而训练集
    `sg_train_a.pt` 是 99.5 GB 且官方 Drive 目录不可续传 —— 本机拿不到。
    所以这里走两条**都能在 500 个真值测试样本上被反向验证**的路线：

    路线 A｜参数扰动 MC（输入侧地质不确定性）
        对 12 通道输入场按物理语义加空间相关噪声（对数渗透率、孔隙度、注入速率、Swi、流度比），
        同一颗权重推理 K 次 → 得到逐格 std。含义是"参数不确定 → 预测场不确定"，
        这正是封存选址风险评估要的那个量，且只花推理钱（GPU 0.2 s/次）。
        ⚠️ 它**不**代表模型权重不确定性，别和 deep ensemble 混为一谈。

    路线 B｜保形校准（distribution-free，有限样本覆盖保证）
        split conformal（Lei et al. 2018）：用校准集上的误差分位数把任意 band 变成
        有覆盖保证的区间 —— P(真值 ∈ 区间) ≥ 1−α 不依赖误差分布假设。
        做法：band = pred ± κ·std_A，在校准集上搜**最小**的 κ 使经验 PICP ≥ 1−α；
        Mondrian 分组（按羽流规模三分位）让不同规模的场地有不同宽度。
        再在一个**没参与 κ 选择**的验证子集上报告 PICP/MPIW —— 这是覆盖率是否真达标的证据。

    指标级区间（给三级预警用）：对每个标量风险指标（突破时间、羽流面积比、侧向越界占比）
    以 |指标(pred) − 指标(y)| 为非一致性分数，保形分位数直接给出 ±q 区间。

自检：`python src/uq_model.py`（合成数据，验证覆盖保证与扰动语义）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

H, W, T, C = 96, 200, 24, 12

# 通道语义（SimpleBlock3d 注释 + 本机实测范围核对）：9/10/11 是网格三步长，属坐标不是参数，永不扰动
CH_LOG_KR, CH_LOG_KZ, CH_POROSITY = 0, 1, 2
CH_INJ_LOC, CH_INJ_RATE = 3, 4
CH_TEMP, CH_PRESSURE, CH_SWI, CH_LAMBDA = 5, 6, 7, 8
GRID_CHANNELS = (9, 10, 11)


@dataclass
class PerturbConfig:
    """参数扰动幅度；σ 用"该通道物理量纲"表述，默认值刻意保守（个位数百分比）。"""
    log_perm_sigma: float = 0.04    # ch0/ch1: kr *= exp(15·σ·z) → 约 ±6%
    porosity_sigma: float = 0.010   # ch2: 绝对加噪（孔隙度 0.05–0.35）
    inj_rate_rel: float = 0.05      # ch4: 乘性 ±5%
    swi_sigma: float = 0.010        # ch7
    lambda_sigma: float = 0.020     # ch8
    corr_len_cells: int = 8         # 噪声空间相关长度（格），模拟非均质连续带而非白噪
    seed: int = 20261030

    def __post_init__(self):
        if self.corr_len_cells < 1:
            raise ValueError("corr_len_cells 必须 ≥ 1")


# --------------------------------------------------------------------------- #
# 路线 A：输入侧参数扰动
# --------------------------------------------------------------------------- #
def _correlated_noise(shape: Tuple[int, int], corr_len: int, rng: np.random.Generator) -> np.ndarray:
    """先在小网格上取白噪、再上采样到 (H,W)，得到的场相关长度≈corr_len；方差归一。"""
    ny, nx = shape
    gy = max(1, int(round(ny / corr_len)))
    gx = max(1, int(round(nx / corr_len)))
    coarse = rng.standard_normal((gy, gx))
    ky, kx = int(np.ceil(ny / gy)), int(np.ceil(nx / gx))
    big = np.repeat(np.repeat(coarse, ky, axis=0), kx, axis=1)[:ny, :nx]
    return big / (big.std() + 1e-12)


def perturb_inputs(a: np.ndarray, cfg: PerturbConfig, rng: np.random.Generator) -> np.ndarray:
    """对单个样本的输入场 (H,W,T,C) 加参数噪声；返回新数组，不改原数组。

    关键约束：
      · 只在储层掩膜（ch0 != 0）内扰动 —— 掩膜外的 0 是"该格不属于计算域"的语义，
        一旦扰动就会造出"不存在的储层"，模型输入直接失真；
      · ch3 注入位置、ch5/6 温压、ch9/10/11 网格：视为已知边界条件，不扰动；
      · 垂深/时间方向不新增结构：渗透率与孔隙度用同一张 (H,W) 图（同一块岩石不随时间变），
        注入速率用一条 (T,) 曲线（同一口井的速率不确定是整体的）。
    """
    a = np.asarray(a, dtype=np.float32)
    if a.shape[-1] != C:
        raise ValueError(f"输入场最后一维应为 {C} 通道，实得 {a.shape}")
    out = a.copy()
    rock = (a[..., CH_LOG_KR] != 0)  # (H,W,T)；掩膜沿时间不变

    for ch, sigma in ((CH_LOG_KR, cfg.log_perm_sigma), (CH_LOG_KZ, cfg.log_perm_sigma)):
        z2d = _correlated_noise(a.shape[:2], cfg.corr_len_cells, rng)
        z3d = np.broadcast_to(z2d[..., None], a.shape[:3])
        # ch = ln(kr)/15 → 加 σ·z 等价于 kr × exp(15·σ·z)
        out[..., ch] = np.where(rock, a[..., ch] + sigma * z3d, a[..., ch])

    z2d = _correlated_noise(a.shape[:2], cfg.corr_len_cells, rng)
    out[..., CH_POROSITY] = np.where(rock, a[..., CH_POROSITY] + cfg.porosity_sigma * z2d[..., None],
                                     a[..., CH_POROSITY])

    zt = rng.standard_normal(T)
    for ch, rel, multiplicative in ((CH_INJ_RATE, cfg.inj_rate_rel, True),
                                    (CH_SWI, cfg.swi_sigma, False),
                                    (CH_LAMBDA, cfg.lambda_sigma, False)):
        b = a[..., ch]
        if multiplicative:
            out[..., ch] = np.where(rock, b * (1.0 + rel * zt[None, None, :]), b)
        else:
            out[..., ch] = np.where(rock, b + rel * zt[None, None, :], b)
    return out.astype(np.float32)


def mc_band(replicas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(K,H,W,T) 副本 → (mean, std)。std 用无偏估计（ddof=1），K=1 时为 0。"""
    r = np.asarray(replicas, dtype=np.float64)
    if r.ndim != 4:
        raise ValueError(f"副本应为 (K,H,W,T)，实得 {r.shape}")
    mean = r.mean(axis=0).astype(np.float32)
    if r.shape[0] > 1:
        std = r.std(axis=0, ddof=1).astype(np.float32)
    else:
        std = np.zeros(r.shape[1:], dtype=np.float32)
    return mean, std


# --------------------------------------------------------------------------- #
# 路线 B：保形（conformal）
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """split conformal 的有限样本分位数：level = ceil((n+1)(1-α))/n，截到 ≤1。

    返回的 q 满足：对新样本，区间 [ŷ−q, ŷ+q] 覆盖真值的概率 ≥ 1−α（可交换性假设下）。
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("没有校准分数，无法做保形校准")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha 应在 (0,1)，实得 {alpha}")
    n = s.size
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s, level))


def mondrian_quantiles(scores: np.ndarray, groups: np.ndarray, alpha: float) -> Dict[int, float]:
    """按组各算一个保形分位数（Mondrian conformal）→ 组内覆盖率各自有保证。"""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if scores.size != groups.size:
        raise ValueError("scores 与 groups 长度不一致")
    out: Dict[int, float] = {}
    for g in np.unique(groups):
        sel = groups == g
        # 组太小则回退全局分位数，避免 ceil 后直接取 max 导致区间爆宽
        out[int(g)] = conformal_quantile(scores[sel], alpha) if sel.sum() >= 20 else conformal_quantile(scores, alpha)
    return out


def band_coverage(mean: np.ndarray, std: np.ndarray, truth: np.ndarray,
                  mask: np.ndarray, kappa: float, min_band: float = 0.0) -> Tuple[float, float]:
    """在掩膜格点上算逐格 PICP（覆盖率）与 MPIW（平均区间宽度）。

    PICP 是"真值落在 pred ± κ·max(std, min_band) 里的格点占比"，逐格而非逐样本 —— 因为
    band 本身就是逐格的；min_band 必须与算 κ 时用的同一个值，否则 κ 的口径就前后不一致。
    MPIW 用同一点集（宽区间必然覆盖高，两个数要一起看）。
    """
    inside = np.asarray(mask, dtype=bool)
    if not inside.any():
        return float("nan"), float("nan")
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), min_band)
    truth = np.asarray(truth, dtype=np.float64)
    covered = (truth[inside] >= (mean - kappa * std)[inside]) & (truth[inside] <= (mean + kappa * std)[inside])
    return float(covered.mean()), float((2 * kappa * std[inside]).mean())


def calibrate_kappa(cal_mean: List[np.ndarray], cal_std: List[np.ndarray],
                    cal_truth: List[np.ndarray], cal_mask: List[np.ndarray],
                    alpha: float, grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """严格版：搜最小 κ 使**每个场地**的经验 PICP 平均起来 ≥ 1−α；返回 (κ, 该 κ 下的 PICP)。

    grid 上限内都达不到目标 → 返回最大 κ 与其 PICP（调用方需报告"未达标"，别假装有保证）。
    """
    if grid is None:
        grid = np.round(np.concatenate([np.arange(0.5, 10.01, 0.25), np.arange(11, 60.1, 1.0)]), 3)
    best_k, best_p = float(grid[-1]), -1.0
    for k in grid:
        ps = [band_coverage(m, s, t, mk, k)[0] for m, s, t, mk in zip(cal_mean, cal_std, cal_truth, cal_mask)]
        p = float(np.nanmean(ps))
        if p > best_p:
            best_k, best_p = float(k), p
        if p >= 1.0 - alpha:
            return float(k), p
    return best_k, best_p


def pooled_kappa(cal_mean: List[np.ndarray], cal_std: List[np.ndarray],
                 cal_truth: List[np.ndarray], cal_mask: List[np.ndarray],
                 alpha: float, min_band: float = 0.01,
                 max_cells_per_site: int = 20000, seed: int = 0) -> float:
    """边际版（推荐当主口径）：把校准集所有掩膜格点的 |误差|/band 汇成一个分数池，
    取保形分位数当 κ —— 保证的是"随机一个格点被覆盖的概率 ≥ 1−α"。

    min_band：给 std 设下限（默认 0.01，即 1% 饱和度）。没有它，干格点上 std≈1e-6 会把
    比值炸到几千，κ 完全由数值噪声决定；设下限等于明确声明"我们不宣称优于 1% 的精度"。
    格点太多就等间隔抽样（同种子可复现），池子有 10⁶ 量级时抽样不影响分位数精度。
    """
    if min_band <= 0:
        raise ValueError("min_band 必须 > 0")
    rng = np.random.default_rng(seed)
    pool = []
    for m, s, t, mk in zip(cal_mean, cal_std, cal_truth, cal_mask):
        inside = np.asarray(mk, dtype=bool)
        band = np.maximum(np.asarray(s, dtype=np.float64)[inside], min_band)
        r = np.abs(np.asarray(t, dtype=np.float64)[inside] - np.asarray(m, dtype=np.float64)[inside]) / band
        if r.size > max_cells_per_site:
            idx = rng.choice(r.size, max_cells_per_site, replace=False)
            r = r[idx]
        pool.append(r)
    return conformal_quantile(np.concatenate(pool), alpha)


def score_intervals(cal_scores: np.ndarray, eval_scores: np.ndarray, alpha: float,
                    cal_groups: Optional[np.ndarray] = None,
                    eval_groups: Optional[np.ndarray] = None) -> Dict[str, object]:
    """已算好的非一致性分数 → 保形半径 + 验证集覆盖率。

    cal_scores 用来选 q；eval_scores 只用来检查 q 是否真兑现（两者必须不重叠）。
    """
    s_cal = np.abs(np.asarray(cal_scores, dtype=np.float64))
    s_eval = np.abs(np.asarray(eval_scores, dtype=np.float64))
    if s_cal.size == 0:
        raise ValueError("校准集为空")
    if cal_groups is None:
        q = conformal_quantile(s_cal, alpha)
        per_group: Dict[int, float] = {}
        picp = float((s_eval <= q + 1e-12).mean()) if s_eval.size else float("nan")
    else:
        cg = np.asarray(cal_groups, dtype=int).ravel()
        eg = np.asarray(eval_groups, dtype=int).ravel()
        if cg.size != s_cal.size or eg.size != s_eval.size:
            raise ValueError("分组标签数量必须与样本数一致")
        per_group = mondrian_quantiles(s_cal, cg, alpha)
        fallback = conformal_quantile(s_cal, alpha)
        q_eval = np.array([per_group.get(int(g), fallback) for g in eg])
        q = float(np.mean(list(per_group.values())))
        picp = float((s_eval <= q_eval + 1e-12).mean()) if s_eval.size else float("nan")
    return {"q": float(q), "q_per_group": per_group, "picp_eval": picp,
            "width_eval": float(2 * q), "n_cal": int(s_cal.size), "n_eval": int(s_eval.size)}


def indicator_intervals(cal_pred: np.ndarray, cal_truth: np.ndarray,
                        eval_pred: np.ndarray, eval_truth: np.ndarray,
                        alpha: float,
                        cal_groups: Optional[np.ndarray] = None,
                        eval_groups: Optional[np.ndarray] = None) -> Dict[str, object]:
    """标量指标的保形区间：分数 = |指标(pred) − 指标(y)|。

    传 cal_groups/eval_groups 即 Mondrian 版：每组各用一个分位数，组内覆盖各自有保证
    （不同规模场地的误差尺度差很多，全局一个 q 会让小场地区间虚宽、大场地覆盖不足）。
    """
    s_cal = np.abs(np.asarray(cal_pred, dtype=np.float64) - np.asarray(cal_truth, dtype=np.float64))
    s_eval = np.abs(np.asarray(eval_pred, dtype=np.float64) - np.asarray(eval_truth, dtype=np.float64))
    return score_intervals(s_cal, s_eval, alpha, cal_groups, eval_groups)


# --------------------------------------------------------------------------- #
# 自检：不依赖数据集，纯合成
# --------------------------------------------------------------------------- #
def _selftest() -> bool:
    rng = np.random.default_rng(7)
    alpha = 0.10

    # 1) 保形覆盖保证：ŷ~N(y,1)，验证 1−α 覆盖率兑现
    y_true = rng.standard_normal(400)
    err = rng.standard_normal(400)
    q = conformal_quantile(np.abs(err), alpha)
    heldout_err = np.abs(rng.standard_normal(4000))
    cov = float((heldout_err <= q).mean())
    assert cov >= 1 - alpha - 0.02, f"保形覆盖率 {cov:.3f} 低于名义 {1-alpha}"

    # 2) κ 校准：合成 std 与真实误差尺度脱钩（低估 3 倍），κ 应把它拉回达标
    n, cells = 30, 500
    means = [np.zeros(cells) for _ in range(n)]
    stds = [np.full(cells, 0.3) for _ in range(n)]
    truths = [rng.standard_normal(cells) for _ in range(n)]
    masks = [np.ones(cells, dtype=bool) for _ in range(n)]
    k, picp = calibrate_kappa(means[:20], stds[:20], truths[:20], masks[:20], alpha)
    assert picp >= 1 - alpha - 1e-9, f"κ 校准后 PICP={picp:.3f} 未达 {1-alpha}"
    assert k > 1.0, f"std 低估时 κ 应 >1，实得 {k}"
    held = float(np.mean([band_coverage(means[j], stds[j], truths[j], masks[j], k)[0]
                          for j in range(20, n)]))
    assert held >= 1 - alpha - 0.05, f"严格版 κ 在未参与选 κ 的场地覆盖 {held:.3f} 崩了"

    # 2b) 边际（pooled）版 κ：给 std 设下限，在未参与选 κ 的场地上验覆盖
    mb = 0.3
    k2 = pooled_kappa(means[:20], stds[:20], truths[:20], masks[:20], alpha, min_band=mb)
    held2 = float(np.mean([band_coverage(means[j], stds[j], truths[j], masks[j], k2, min_band=mb)[0]
                           for j in range(20, n)]))
    assert held2 >= 1 - alpha - 0.05, f"pooled κ 未兑现覆盖：{held2:.3f}（κ={k2:.2f}）"
    assert k2 > 1.0, "std 低估 3 倍时 pooled κ 也应 >1"

    # 3) 扰动语义：掩膜外不许被造出储层；网格通道不动；同种子可复现
    a = np.zeros((H, W, T, C), np.float32)
    a[:52, :, :, CH_LOG_KR] = 0.4
    a[:52, :, :, CH_LOG_KZ] = 0.3
    a[:52, :, :, CH_POROSITY] = 0.18
    a[:52, :, :, CH_INJ_RATE] = 0.5
    a[:52, :, :, CH_SWI] = 0.2
    a[:52, :, :, CH_LAMBDA] = 0.5
    a[..., 9], a[..., 10], a[..., 11] = 1.0, 2.0, 3.0
    cfg = PerturbConfig()
    p1 = perturb_inputs(a, cfg, np.random.default_rng(cfg.seed))
    p2 = perturb_inputs(a, cfg, np.random.default_rng(cfg.seed))
    assert np.array_equal(p1, p2), "同种子扰动必须逐位可复现"
    assert np.all(p1[52:, :, :, CH_LOG_KR] == 0), "掩膜外被造出了储层"
    assert np.array_equal(p1[..., GRID_CHANNELS], a[..., GRID_CHANNELS]), "网格通道被扰动了"
    assert not np.allclose(p1[..., CH_LOG_KR], a[..., CH_LOG_KR]), "扰动没生效"
    # kr = exp(ch·15)：σ=0.04 → 单格倍率应在 exp(±15·0.04·3)≈0.1~10 的合理带内
    ratio = np.exp(15 * (p1[:52, :, :, CH_LOG_KR] - a[:52, :, :, CH_LOG_KR]))
    assert 0.05 < ratio.min() and ratio.max() < 20, f"渗透率倍率范围异常 [{ratio.min():.3f},{ratio.max():.3f}]"

    # 4) MC band：两副本 std≈1.4142（K=2 无偏），单副本 std 必须为 0
    rep = np.stack([np.ones((2, 2, 2)), np.ones((2, 2, 2)) * 3])
    mn, sd = mc_band(rep)
    assert np.allclose(mn, 2.0), f"均值应全为 2，实得 {mn[0,0,0]}"
    assert abs(float(sd[0, 0, 0]) - 1.4142136) < 1e-4, f"两副本 std 应≈1.4142，实得 {sd[0,0,0]}"
    mn1, sd1 = mc_band(np.ones((1, 2, 2, 2)))
    assert float(sd1.max()) == 0.0, "单副本 std 应为 0"

    # 5) 指标区间：验证子集覆盖率兑现
    yt = rng.standard_normal(200)
    yhat = yt + rng.standard_normal(200) * 0.5
    r = indicator_intervals(yhat[:100], yt[:100], yhat[100:], yt[100:], 0.10)
    assert r["picp_eval"] >= 1 - alpha - 0.03, f"指标级覆盖率 {r['picp_eval']:.3f} 未达名义"

    print("[uq_model.selftest] 通过：保形覆盖兑现 / κ 校准达标 / 扰动守掩膜与网格通道且可复现 / MC band 正确。")
    return True


if __name__ == "__main__":
    _selftest()
