"""
test_uq_risk.py —— UQ 引擎与三级预警的算法健全性测试
====================================================
运行：.venv/Scripts/python.exe -m pytest tests/test_uq_risk.py -q
只依赖 numpy：不加载权重、不做推理，校验的是"算法本身有没有说谎"。
里面有几条是针对实测踩过的坑写的回归测试，注释里标了坑名。
"""
import numpy as np

from src.risk_model import (RiskLevel, RiskReference, SiteIndicators, TierPolicy,
                            classify_site, composite_score, extract_indicators,
                            interval_tier, physical_time_days)
from src.uq_model import (CH_INJ_RATE, CH_INJ_LOC, CH_LAMBDA, CH_LOG_KR, CH_LOG_KZ, CH_POROSITY,
                          CH_SWI, CH_TEMP, GRID_CHANNELS, PerturbConfig, band_coverage,
                          calibrate_kappa, conformal_quantile, edit_channels, indicator_intervals,
                          mc_band, mondrian_quantiles, pooled_kappa, perturb_inputs,
                          score_intervals)

ALPHA = 0.10
H, W, T, C = 12, 20, 24, 12


# --------------------------------------------------------------------------- #
# 合成数据构造：故意让"干格点准、羽流内差"，用来检验口径选择
# --------------------------------------------------------------------------- #
def make_dry_plume_case(seed: int, n_cell: int = 10, plume_share: float = 0.25):
    """返回 (center, std, truth, mask)。std 是模型自报的不确定性，故意远小于真实误差。"""
    rng = np.random.default_rng(seed)
    truth = np.zeros((n_cell, n_cell))
    n_plume = int(round(plume_share * n_cell * n_cell))
    idx = rng.choice(n_cell * n_cell, n_plume, replace=False)
    truth.ravel()[idx] = rng.uniform(0.3, 0.9, n_plume)
    plume = truth > 0.0
    err = rng.normal(0.0, 0.005, truth.shape)
    err[plume] = rng.normal(0.0, 0.05, int(plume.sum()))     # 羽流内误差大十倍
    std = np.where(plume, 0.002, 0.001)                      # 自报不确定性两者都很小
    return truth + err, std, truth, np.ones(truth.shape, dtype=bool), plume


# --------------------------------------------------------------------------- #
# 保形：有限样本修正 + 覆盖兑现
# --------------------------------------------------------------------------- #
def test_conformal_quantile_finite_sample_correction():
    q = conformal_quantile(np.array([0.0, 1.0]), 0.10)      # n=2, level=ceil(3*0.9)/2=2/2=1
    assert q == 1.0, "样本极少时必须取到最大值，不能靠外推吹覆盖率"
    assert conformal_quantile(np.full(50, 0.7), 0.10) == 0.7
    for alpha in (0.05, 0.10, 0.20):
        assert conformal_quantile(np.arange(100.0), alpha) >= np.quantile(np.arange(100.0), 1 - alpha) - 1e-9


def test_conformal_holdout_coverage_delivered():
    hits = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        cal = np.abs(rng.normal(0, 1, 200))
        ev = np.abs(rng.normal(0, 1, 200))
        r = score_intervals(cal, ev, ALPHA)
        hits.append(r["picp_eval"])
    assert np.mean(hits) >= 1 - ALPHA, f"验证集平均覆盖 {np.mean(hits):.3f} 未兑现 {1-ALPHA}"


def test_score_intervals_rejects_empty_calibration():
    try:
        score_intervals(np.array([]), np.array([0.1]), ALPHA)
    except ValueError:
        return
    raise AssertionError("空校准集应报错，而不是返回一个假的分位数")


def test_mondrian_small_group_falls_back_to_global():
    rng = np.random.default_rng(3)
    scores = np.abs(rng.normal(0, 1, 60))
    groups = np.array([0] * 50 + [1] * 10)
    per = mondrian_quantiles(scores, groups, ALPHA)
    assert per[1] == conformal_quantile(scores, ALPHA), "不足 20 个的组必须回退全局分位"
    assert per[0] == conformal_quantile(scores[groups == 0], ALPHA), "够 20 个的组应用自己的分数"
    assert set(per) == {0, 1}


# --------------------------------------------------------------------------- #
# 逐格 band：单调性、下限、以及"校准口径"这个坑
# --------------------------------------------------------------------------- #
def test_band_coverage_monotone_in_kappa():
    center, std, truth, mask, _ = make_dry_plume_case(0)
    prev_picp, prev_mpiw = -1.0, -1.0
    for k in (0.5, 1, 2, 5, 20):
        picp, mpiw = band_coverage(center, std, truth, mask, k)
        assert picp >= prev_picp - 1e-12, "κ 增大覆盖率不应下降"
        assert mpiw > prev_mpiw, "κ 增大区间必然变宽"
        prev_picp, prev_mpiw = picp, mpiw


def test_min_band_declares_precision_floor():
    center = np.zeros(100)
    std = np.full(100, 1e-9)                                # 模型自报"无限准"
    truth = np.zeros(100)
    mask = np.ones(100, dtype=bool)
    _, mpiw = band_coverage(center, std, truth, mask, 1.0, min_band=0.0)
    assert mpiw < 1e-8, "不设下限时区间宽度就是数值噪声，等于谎报精度"
    _, mpiw = band_coverage(center, std, truth, mask, 1.0, min_band=0.01)
    assert abs(mpiw - 0.02) < 1e-12, "设了 min_band 后宽度应为 2κ·min_band"


def test_reservoir_scope_calibration_under_covers_plume():
    """坑名「干格点稀释」：真值 75% 是零，模型在零上也几乎为零，
    拿全储层格点选 κ 会选出一个很小的 κ，但它对羽流内部毫无保护。
    这条测试就是 100 场地实测里 κ=0.19 时羽流内覆盖只有 0.367 的复现。"""
    cal = [make_dry_plume_case(s) for s in range(24)]
    ev = [make_dry_plume_case(100 + s) for s in range(16)]
    args = lambda cs: ([c[0] for c in cs], [c[1] for c in cs], [c[2] for c in cs], [c[3] for c in cs])
    plume_args = lambda cs: ([c[0] for c in cs], [c[1] for c in cs], [c[2] for c in cs], [c[4] for c in cs])

    k_all = pooled_kappa(*args(cal), ALPHA, min_band=0.01)
    k_pl = pooled_kappa(*args(plume_args(cal)), ALPHA, min_band=0.01)
    assert k_pl > k_all, f"羽流内校准的 κ({k_pl:.2f}) 应大于全储层(κ={k_all:.2f})"

    picp_all_scope = np.mean([band_coverage(*c[:4], k_all, min_band=0.01)[0] for c in ev])
    picp_plume_bad = np.mean([band_coverage(c[0], c[1], c[2], c[4], k_all, min_band=0.01)[0] for c in ev])
    picp_plume_ok = np.mean([band_coverage(c[0], c[1], c[2], c[4], k_pl, min_band=0.01)[0] for c in ev])
    assert picp_all_scope >= 1 - ALPHA - 0.05, \
        f"全储层 κ 至少要在自己的口径上接近达标（否则是选 κ 的 bug），实得 {picp_all_scope:.3f}"
    assert picp_plume_bad < 0.6, f"用全储层 κ 时羽流内覆盖应明显塌陷，实得 {picp_plume_bad:.2f}"
    # 只有 8 个留出场地，经验覆盖会在承诺值上下浮动，所以这里留 0.05 的容差而不是硬卡 0.90
    assert picp_plume_ok >= 1 - ALPHA - 0.05, f"换成羽流内 κ 后羽流内覆盖应兑现，实得 {picp_plume_ok:.2f}"
    assert picp_plume_ok - picp_plume_bad > 0.3, "换口径的收益必须是显著的"


def test_calibrate_kappa_is_stricter_than_pooled():
    """严格口径要求每个场地各自达标，κ 必然 ≥ 边际口径 —— 这是两者关系不反的证明。"""
    cal = [make_dry_plume_case(s) for s in range(12)]
    k_pool = pooled_kappa(*([c[0] for c in cal], [c[1] for c in cal],
                            [c[2] for c in cal], [c[3] for c in cal]), ALPHA, min_band=0.01)
    k_site, picp = calibrate_kappa([c[0] for c in cal], [c[1] for c in cal],
                                   [c[2] for c in cal], [c[3] for c in cal], ALPHA)
    assert k_site >= k_pool, f"严格 κ({k_site}) 不应小于边际 κ({k_pool})"
    assert picp >= 1 - ALPHA or k_site >= 60, "报告严格口径时要同时说明它是否达标"


# --------------------------------------------------------------------------- #
# 参数扰动：可复现、不动边界条件、不动网格通道
# --------------------------------------------------------------------------- #
def _fake_input(seed: int = 0):
    """掩膜外（行 0..1 与 9..11）全通道保持 0 —— 与官方数据同语义：ch0=ln(kr)/15，
    为 0 即"该格不属于计算域"。所以这里绝不能给整网格一个非零基线，否则
    perturb_inputs 会把围岩也当储层扰动。"""
    rng = np.random.default_rng(seed)
    a = np.zeros((H, W, T, C), np.float32)
    rock = np.zeros((H, W, T), bool)
    rock[2:9] = True

    def fill(ch: int, vals) -> None:
        a[..., ch][rock] = np.asarray(vals, np.float32)[rock]

    fill(CH_LOG_KR, np.log(np.abs(rng.normal(0.5, 0.2, (H, W, T)))) / 15.0)
    fill(CH_LOG_KZ, a[..., CH_LOG_KR])
    fill(CH_POROSITY, np.abs(rng.normal(0.2, 0.05, (H, W, T))))
    fill(CH_INJ_RATE, np.abs(rng.normal(1.0, 0.1, (H, W, T))))
    fill(CH_SWI, np.full((H, W, T), 0.25))
    a[..., CH_INJ_LOC] = 0.5
    a[..., CH_TEMP] = 60.0
    for ch in GRID_CHANNELS:                                # 网格三步长：坐标，不是参数
        a[..., ch] = 1.0
    return a


def test_perturbation_is_reproducible():
    a = _fake_input(1)
    cfg = PerturbConfig()
    x1 = perturb_inputs(a, cfg, np.random.default_rng(7))
    x2 = perturb_inputs(a, cfg, np.random.default_rng(7))
    assert np.array_equal(x1, x2), "同种子必须逐位相同，否则 8 副本无法复现"
    x3 = perturb_inputs(a, cfg, np.random.default_rng(8))
    assert not np.array_equal(x1, x3), "换种子应给出不同副本"
    assert x1.shape == a.shape and x1.dtype == a.dtype
    assert not np.array_equal(a, x1), "扰动应真的改动输入，而不是原样返回"


def test_perturbation_respects_structural_channels():
    a = _fake_input(2)
    out = perturb_inputs(a, PerturbConfig(), np.random.default_rng(5))
    rock = a[..., CH_LOG_KR] != 0
    for ch in GRID_CHANNELS + (CH_INJ_LOC, CH_TEMP):
        assert np.array_equal(out[..., ch], a[..., ch]), f"通道 {ch} 是坐标/边界条件，不该被扰动"
    assert np.array_equal(out[~rock], a[~rock]), "掩膜外的 0 表示'不属于计算域'，扰动它就造出假储层"
    ratio = np.where(rock, out[..., CH_INJ_RATE] / np.where(a[..., CH_INJ_RATE] != 0, a[..., CH_INJ_RATE], 1), np.nan)
    per_time = np.nanstd(ratio, axis=(0, 1))
    assert np.allclose(per_time, 0.0, atol=1e-5), "同一时刻整口井的注入速率应同比缩放，各格系数一致"


def test_perturbation_is_spatially_correlated_not_white():
    """同一块岩石的扰动应沿深度/时间保持一致（相关长度 8 格），白噪会造出物理上不存在的毛刺。"""
    a = _fake_input(3)
    cfg = PerturbConfig(corr_len_cells=8)
    out = perturb_inputs(a, cfg, np.random.default_rng(11))
    d = (out[..., CH_LOG_KR] - a[..., CH_LOG_KR])[:, :, 0]  # (H,W) 单时刻
    assert d.std() > 1e-4, "扰动幅度为 0 说明掩膜条件写错了"
    adj = np.abs(d[:, 1:] - d[:, :-1]).mean()
    far = np.abs(d - d[:, ::-1]).mean()
    assert adj < 0.5 * far, f"相邻格应高度相关（adj={adj:.4f} far={far:.4f}）"
    dt = np.abs(out[..., CH_LOG_KR] - a[..., CH_LOG_KR])
    assert np.allclose(dt, dt[:, :, 0:1], atol=1e-6), "渗透率是静态属性，不该随时间变化"
    assert not np.any(dt[2:9] == 0), "储层内每格都该被扰动到"


def test_edit_channels_uses_the_same_units_as_perturbation():
    """交互原型让使用者显式改参数；量纲必须和 perturb_inputs 一致，否则沙盘与跑批说的不是同一种话。"""
    a = _fake_input(4)
    rock = a[..., CH_LOG_KR] != 0
    out = edit_channels(a, perm_mult=1.5, poro_add=0.02, rate_mult=2.0, swi_add=-0.01, lam_add=0.05)
    d = np.log(1.5) / 15.0
    for ch in (CH_LOG_KR, CH_LOG_KZ):
        assert np.allclose((out[..., ch] - a[..., ch])[rock], d), "渗透率 ×1.5 = 对数通道加 ln(1.5)/15"
    assert np.allclose((out[..., CH_POROSITY] - a[..., CH_POROSITY])[rock], 0.02)
    assert np.allclose(out[..., CH_INJ_RATE][rock], a[..., CH_INJ_RATE][rock] * 2.0), "注入速率是乘性"
    assert np.allclose((out[..., CH_SWI] - a[..., CH_SWI])[rock], -0.01)
    assert np.allclose((out[..., CH_LAMBDA] - a[..., CH_LAMBDA])[rock], 0.05)
    for ch in GRID_CHANNELS + (CH_INJ_LOC, CH_TEMP):
        assert np.array_equal(out[..., ch], a[..., ch]), f"通道 {ch} 是坐标/边界条件，不接受编辑"
    assert np.array_equal(out[~rock], a[~rock]), "掩膜外一动就造出假储层"
    assert np.array_equal(edit_channels(a), a), "全默认值必须原样返回"


def test_edit_channels_validates_inputs():
    a = _fake_input(5)
    for bad in (0.0, -1.0):
        try:
            edit_channels(a, perm_mult=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"倍率 {bad} 应被拒绝（ln 无意义）")


def test_mc_band_uses_unbiased_std():
    r = np.random.default_rng(0).normal(size=(5, 4, 4, 3))
    mean, std = mc_band(r)
    assert np.allclose(mean, r.mean(0), rtol=1e-6)
    assert np.allclose(std, r.std(0, ddof=1), rtol=1e-5), "副本少时用有偏 std 会系统性低估不确定性"
    _, std1 = mc_band(r[:1])
    assert np.all(std1 == 0), "K=1 没有离散度可谈，必须返回 0 而不是报错或假装有值"


# --------------------------------------------------------------------------- #
# 指标提取
# --------------------------------------------------------------------------- #
def test_physical_time_days_series():
    d = physical_time_days(24)
    assert d[0] == 1.0 and np.all(np.diff(d) > 0) and d.size == 24


def _tiny_field(with_top_hit: bool, outside: bool = False, over_one: bool = False):
    """10×20×5 的小场地：掩膜行 3..8 是储层，行 3 即"储层顶"。
    羽流放列 8..11 —— 默认的"边界列"是两侧各 5 列（0..4 与 15..19），必须避开，
    否则侧向占比就不是 0 了。"""
    sg = np.zeros((10, 20, 5), np.float32)
    m = np.zeros((10, 20), bool)
    m[3:9, :] = True
    sg[5:7, 8:12, :] = 0.4                                  # 储层中部的羽流
    if with_top_hit:
        sg[3, 8:10, 2:] = 0.5                               # 第 2 帧起顶到储层顶行
    if outside:
        sg[0, 8:10, 3:] = 0.3                               # 掩膜外信号（模型伪影）
    if over_one:
        sg[6, 10, 4] = 1.7                                  # 饱和度 > 1，物理违背
    return sg, m


def test_extract_indicators_breakthrough_and_absence():
    days = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    ind = extract_indicators(*_tiny_field(True), days=days)  # type: ignore[misc]
    assert ind.breakthrough_days == 4.0, f"顶行首次出现应在第 2 帧=4 天，实得 {ind.breakthrough_days}"
    assert ind.ratio_peak > 0 and ind.ratio_mean > 0
    assert ind.outside_share == 0.0 and ind.violation_l1 < 1e-6

    never = extract_indicators(*_tiny_field(False), days=days)  # type: ignore[misc]
    assert np.isnan(never.breakthrough_days), "从未顶到储层顶行时突破时间必须是 NaN（最安全），不能是 0"


def test_extract_indicators_artifacts_are_separate_axes():
    days = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    sg, m = _tiny_field(True, outside=True, over_one=True)
    ind = extract_indicators(sg, m, days=days)
    assert ind.outside_share > 0, "掩膜外伪影要能被量化 —— 它是可信度轴，不是风险轴"
    assert ind.violation_share > 0 and ind.violation_l1 > 0
    assert ind.lateral_share == 0.0, "羽流没贴边界列，侧向占比应为 0"


# --------------------------------------------------------------------------- #
# 相对分级：阈值比的是"队列分位"，不是加权和的绝对值
# --------------------------------------------------------------------------- #
def _anti_correlated_cohort(n: int = 20):
    """突破越早 ↔ 面积越大 被刻意做成反向，使加权和挤在 0.35–0.45，
    复现真值队列的退化情形（实测 100 场地综合分最大 0.82，按绝对 0.90 判则永远没有红）。"""
    return [SiteIndicators(float(100 + i), 0.1 + 0.01 * i, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0)
            for i in range(n)]


def test_tier_threshold_is_cohort_percentile_not_absolute():
    pol = TierPolicy()
    ref = RiskReference(_anti_correlated_cohort())
    scores = ref.score_cohort(pol)
    assert scores.max() < pol.red_pct, "前提：加权和的绝对值最高也够不到红线"
    worst = classify_site(ref.inds[0], ref, pol)            # i=0：突破最早、面积最小 → 综合分最高
    assert worst.level == RiskLevel.RED, f"队列里最差的那个必须能判红，实得 {worst.level}"
    assert worst.score >= pol.red_pct
    best = classify_site(ref.inds[-1], ref, pol)
    assert best.level == RiskLevel.GREEN and best.score < pol.yellow_pct
    mid = classify_site(ref.inds[10], ref, pol)
    assert mid.level is not RiskLevel.RED, "队列中段的场地不该被判红"


def test_composite_score_is_normalized_by_weights():
    pol = TierPolicy(weights={"breakthrough": 45.0, "areal": 35.0, "lateral": 20.0})
    assert abs(sum(pol.weights.values()) - 1.0) < 1e-12
    assert abs(composite_score({"breakthrough": 1.0, "areal": 1.0, "lateral": 1.0}, pol) - 1.0) < 1e-12


def test_degenerate_reference_does_not_flag_everyone():
    """坑名「退化队列人人并列」：真值的掩膜外占比恒为 0，用它当参照会把任何非零值算成最差，
    于是每个场地都被"可信度不足"降级、全变黄。"""
    pol = TierPolicy()
    degen = RiskReference([SiteIndicators(100.0, 0.3, 0.1, 0.05, 0.1, 0.0, 0.0, 0.0) for _ in range(24)])
    j = classify_site(SiteIndicators(float("nan"), 0.01, 0.005, 0.0, 0.001, 0.24, 0.05, 0.01),
                      degen, pol, conf_ref=degen)
    assert j.level == RiskLevel.GREEN, f"退化参照不该触发降级，实得 {j.level}"
    assert degen.confidence_percentile("outside_share", 0.5) == 0.0


def test_low_confidence_blocks_green():
    pol = TierPolicy()
    ref = RiskReference([SiteIndicators(100.0, 0.3, 0.1, 0.05, 0.1, o / 100.0, 0.0, 0.0) for o in range(24)])
    j = classify_site(SiteIndicators(float("nan"), 0.01, 0.005, 0.0, 0.001, 0.24, 0.0, 0.0), ref, pol)
    assert j.level == RiskLevel.YELLOW, "几何上像绿，但伪影处于队列最差 10% → 不得判绿"
    assert any("可信度" in r for r in j.reasons)


def test_interval_tier_three_branches():
    pol = TierPolicy()
    assert interval_tier(0.92, 0.99, pol) == RiskLevel.RED
    assert interval_tier(0.30, 0.70, pol) == RiskLevel.GREEN
    assert interval_tier(0.60, 0.95, pol) == RiskLevel.YELLOW
    assert interval_tier(0.76, 0.95, pol) == RiskLevel.YELLOW, "点判红但下界不足红线 → 说不准就是黄"


def _clear_outlier_cohort(n: int = 100):
    """队列里第 0 个场地三根轴全都最差，与其余场地之间留出大gap。
    保形半径必须"小到不足以改变结论时保住结论"，靠的就是这个 gap ——
    若队列分数密集（见 _anti_correlated_cohort），挪一格分位就会改变判级，那正是黄档该出现的场合。"""
    others = [SiteIndicators(10.0 + 5.0 * i, 0.30 - 0.002 * i, 0.1, 0.0, 0.02, 0.0, 0.0, 0.0)
              for i in range(1, n)]
    return [SiteIndicators(1.0, 0.60, 0.4, 0.2, 0.60, 0.0, 0.0, 0.0)] + others


def test_wider_conformal_radius_moves_sites_into_yellow():
    """UQ 改变结论的证据：区间变宽时，绿不会变红、红不会变绿，只会汇进黄档。"""
    pol = TierPolicy()
    ref = RiskReference(_clear_outlier_cohort())
    tight = classify_site(ref.inds[0], ref, pol, q={"breakthrough": 0.5, "areal": 0.001, "lateral": 0.001})
    wide = classify_site(ref.inds[0], ref, pol, q={"breakthrough": 300.0, "areal": 0.3, "lateral": 0.61})
    assert tight.level == RiskLevel.RED, f"半径≈0 时应回到点判结论，实得 {tight.level}"
    assert wide.level == RiskLevel.YELLOW, f"半径吃掉 gap 后应改判黄，实得 {wide.level}"
    assert wide.score_lo <= wide.score_hi


def test_interval_bounds_are_ordered_for_every_site():
    pol = TierPolicy()
    ref = RiskReference(_anti_correlated_cohort())
    q = {"breakthrough": 20.0, "areal": 0.02, "lateral": 0.02}
    for ind in ref.inds:
        c = classify_site(ind, ref, pol, q=q)
        assert c.score_lo <= c.score <= c.score_hi, f"区间应包住点估计，实得 {c.score_lo:.2f}/{c.score:.2f}/{c.score_hi:.2f}"


def test_indicator_intervals_report_their_own_coverage():
    """半径必须在"没参与选 q"的样本上验证，否则覆盖率是自己给自己打分。"""
    rng = np.random.default_rng(7)
    cal_pred = rng.normal(0, 1, 120)
    cal_truth = cal_pred + rng.normal(0, 0.3, 120)
    ev_pred = rng.normal(0, 1, 120)
    ev_truth = ev_pred + rng.normal(0, 0.3, 120)
    r = indicator_intervals(cal_pred, cal_truth, ev_pred, ev_truth, ALPHA)
    assert r["q"] > 0 and r["picp_eval"] >= 1 - ALPHA - 0.05
    assert r["n_cal"] == 120 and r["n_eval"] == 120
    assert abs(r["width_eval"] - 2 * r["q"]) < 1e-9, "宽度必须和半径自洽（PICP 和 MPIW 要一起报）"
