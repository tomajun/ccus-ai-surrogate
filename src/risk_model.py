"""
risk_model.py —— 三级泄漏风险预警引擎（几何感知 + 6 机制映射 + UQ 驱动）
=========================================================================

⚠️ 诚实声明（必读）：
    本模块是一套**启发式筛查规则**，不是经过验证的"风险认证模型"。
    - 它不依赖任何训练数据，纯几何 + 阈值逻辑，**现在就能跑、能自检**。
    - 但它**无法被验证对错**：CO2 封存领域没有公开可用的"是否真实泄漏"标签
      来校验这些阈值。阈值（bands / radius / uncertainty_threshold）都是
      *人为设定、待真实羽流统计校准* 的占位值。
    - 它必须接在一个**能跑出真实羽流的预测器**（U-FNO 复现后）之上才有意义；
      单独运行只能证明"逻辑会触发"，不能证明"报得对"。
    → 大创答辩定位：写清"这是不确定性感知的启发式预警筛查"，并写明局限与待校准项。

与 src/models/risk_classifier.py（阶段 A 玩具 RiskMLP）的区别：
    - 旧 RiskMLP：3 参数 + 5 个场统计 → 3 类，纯数据拟合，无地质几何、无 UQ、无机制映射，
      且建立在合成死路数据上，**已判定为不可用**。
    - 本 risk_model：输入"羽流场 + 不确定性场 + 地质几何"，逐条对照 6 种泄漏机制
      给出绿/黄/红 + 触发原因 + 命中机制编号，可直接吃 U-FNO 输出。

六种泄漏机制（编号与 docs/项目组内介绍 文档中的剖面图一一对应）：
    1 盖层密封失效     plume 进入盖层带（顶部 band）
    2 废弃井筒泄漏     plume 触及废弃井位置
    3 断层活化泄漏     plume / 不确定性带 跨越断层段
    4 侧向越界运移     plume 触及储层四周边界
    5 溶解/卤水入侵    plume 逼近储层底界（下方含水层）
    6 近地表/地表渗漏  plume 突破至极浅层（比盖层更靠顶）

接口：
    cfg = RiskConfig(wells=[(0.5,0.1)], faults=[((0.2,0.0),(0.4,0.6))])
    res = assess(plume_field, uncertainty_field=None, config=cfg)
    res.level            # RiskLevel.GREEN / YELLOW / RED
    res.mechanism_ids    # 命中的机制编号列表，例如 [2]
    res.reasons          # 人读的解释列表
    res.metrics          # 各触发距离等诊断量
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class RiskLevel(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2

    @property
    def label(self) -> str:
        return {0: "绿（安全圈闭）", 1: "黄（需关注）", 2: "红（高风险）"}[self.value]


MECHANISM_NAMES = {
    1: "盖层密封失效",
    2: "废弃井筒泄漏",
    3: "断层活化泄漏",
    4: "侧向越界运移",
    5: "溶解/卤水入侵含水层",
    6: "近地表/地表渗漏",
}


@dataclass
class RiskConfig:
    """所有阈值均为占位默认值，需接真实羽流统计后校准。坐标归一化到 [0,1]。"""
    # 几何：井（点）与断层（线段），归一化坐标
    wells: List[Tuple[float, float]] = field(default_factory=list)
    faults: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)

    # 顶部 band 比例（y 从 0=顶 到 1=底）
    near_surface_frac: float = 0.03   # 极浅层 → 机制 6
    caprock_frac: float = 0.10        # 盖层带 → 机制 1（需 > near_surface_frac）
    aquifer_frac: float = 0.10        # 底界带 → 机制 5（y > 1-aquifer_frac）
    boundary_frac: float = 0.05       # 左右边界带 → 机制 4

    # 触发半径（归一化）
    well_radius_frac: float = 0.05     # 井筒触发半径
    fault_radius_frac: float = 0.05    # 断层触发半径

    # 黄色预警：羽流接近但未进入 band/结构 的容差距离
    yellow_proximity_frac: float = 0.08

    # 羽流识别
    sat_threshold: float = 0.01        # 判定为羽流的饱和度阈值

    # UQ 驱动预警：不确定性均值超过此值 → 至少黄
    uncertainty_threshold: float = 0.05
    uncertainty_k: float = 1.0         # 不确定性带 = plume + k*std


@dataclass
class RiskResult:
    level: RiskLevel
    mechanism_ids: List[int]
    reasons: List[str]
    metrics: dict

    def __str__(self) -> str:
        mechs = "、".join(MECHANISM_NAMES[m] for m in self.mechanism_ids) or "无"
        return (f"[{self.level.label}] 命中机制：{mechs}\n"
                + "\n".join(f"  - {r}" for r in self.reasons))


# --------------------------------------------------------------------------- #
# 几何距离工具（归一化坐标）
# --------------------------------------------------------------------------- #
def _plume_pixels(plume: np.ndarray, thr: float):
    """返回羽流像素的归一化坐标 (Ys, Xs)。支持 2D(H,W) 或 3D(T,H,W)（取末帧）。"""
    if plume.ndim == 3:
        field = plume[-1]
    else:
        field = plume
    field = np.asarray(field, dtype=np.float32)
    H, W = field.shape
    mask = field > thr
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0,)), np.empty((0,)), H, W, field
    Ys = (ys + 0.5) / H
    Xs = (xs + 0.5) / W
    return Ys, Xs, H, W, field


def _point_dist(Xs: np.ndarray, Ys: np.ndarray, px: float, py: float) -> float:
    if Xs.size == 0:
        return float("inf")
    return float(np.min(np.sqrt((Xs - px) ** 2 + (Ys - py) ** 2)))


def _seg_dist(Xs: np.ndarray, Ys: np.ndarray,
              x1: float, y1: float, x2: float, y2: float) -> float:
    """点到线段最近距离（批量到所有羽流像素取最小）。"""
    if Xs.size == 0:
        return float("inf")
    vx, vy = x2 - x1, y2 - y1
    seg_len2 = vx ** 2 + vy ** 2
    if seg_len2 < 1e-12:
        return _point_dist(Xs, Ys, x1, y1)
    t = ((Xs - x1) * vx + (Ys - y1) * vy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    cx, cy = x1 + t * vx, y1 + t * vy
    d = np.sqrt((Xs - cx) ** 2 + (Ys - cy) ** 2)
    return float(np.min(d))


# --------------------------------------------------------------------------- #
# 主评估函数（每条触发显式携带 severity，避免文本解析定级的脆弱性）
# --------------------------------------------------------------------------- #
def assess(plume: np.ndarray,
           uncertainty: Optional[np.ndarray] = None,
           config: Optional[RiskConfig] = None) -> RiskResult:
    cfg = config or RiskConfig()
    Ys, Xs, H, W, field = _plume_pixels(plume, cfg.sat_threshold)

    # triggers: list of (mech_id, severity, reason)
    triggers: List[Tuple[int, RiskLevel, str]] = []
    metrics: dict = {"plume_fraction": float((field > cfg.sat_threshold).mean()),
                     "plume_max": float(field.max()),
                     "has_plume": bool(Ys.size > 0)}

    # ---- 机制 1 / 6：顶部 band ----
    if Ys.size > 0:
        top_min = float(Ys.min())
        if top_min < cfg.near_surface_frac:
            triggers.append((6, RiskLevel.RED,
                             f"羽流突破至极浅层（top_y={top_min:.3f} < {cfg.near_surface_frac}），"
                             f"近地表/地表渗漏高风险（机制6）。"))
        elif top_min < cfg.caprock_frac:
            triggers.append((1, RiskLevel.RED,
                             f"羽流进入盖层带（top_y={top_min:.3f} < {cfg.caprock_frac}），"
                             f"盖层密封失效高风险（机制1）。"))
        elif top_min < cfg.caprock_frac + cfg.yellow_proximity_frac:
            triggers.append((1, RiskLevel.YELLOW,
                             f"羽流接近盖层（top_y={top_min:.3f}），存在上侵趋势，列入关注（机制1预警）。"))

    # ---- 机制 5：底界 band ----
    if Ys.size > 0:
        bot_max = float(Ys.max())
        if bot_max > 1.0 - cfg.aquifer_frac:
            triggers.append((5, RiskLevel.RED,
                             f"羽流逼近储层底界（bot_y={bot_max:.3f} > {1-cfg.aquifer_frac:.3f}），"
                             f"向下入侵含水层高风险（机制5）。"))
        elif bot_max > 1.0 - cfg.aquifer_frac - cfg.yellow_proximity_frac:
            triggers.append((5, RiskLevel.YELLOW,
                             f"羽流接近底界（bot_y={bot_max:.3f}），关注向下运移（机制5预警）。"))

    # ---- 机制 4：侧向边界 ----
    if Xs.size > 0:
        left_min, right_max = float(Xs.min()), float(Xs.max())
        if left_min < cfg.boundary_frac or right_max > 1.0 - cfg.boundary_frac:
            triggers.append((4, RiskLevel.RED,
                             f"羽流触及储层侧向边界（x∈[{left_min:.3f},{right_max:.3f}]），"
                             f"越界运移高风险（机制4）。"))
        elif (left_min < cfg.boundary_frac + cfg.yellow_proximity_frac
              or right_max > 1.0 - cfg.boundary_frac - cfg.yellow_proximity_frac):
            triggers.append((4, RiskLevel.YELLOW,
                             f"羽流接近侧向边界（x∈[{left_min:.3f},{right_max:.3f}]），"
                             f"关注越界（机制4预警）。"))

    # ---- 机制 2：废弃井筒（点） ----
    for i, (wx, wy) in enumerate(cfg.wells):
        d = _point_dist(Xs, Ys, wx, wy)
        metrics[f"well_{i}_dist"] = d
        if d < cfg.well_radius_frac:
            triggers.append((2, RiskLevel.RED,
                             f"羽流触及废弃井筒 #{i}（距 {d:.3f} < {cfg.well_radius_frac}），"
                             f"井筒泄漏高风险（机制2）。"))
        elif d < cfg.well_radius_frac + cfg.yellow_proximity_frac:
            triggers.append((2, RiskLevel.YELLOW,
                             f"羽流接近废弃井筒 #{i}（距 {d:.3f}），关注井筒泄漏（机制2预警）。"))

    # ---- 机制 3：断层（线段） ----
    for i, ((x1, y1), (x2, y2)) in enumerate(cfg.faults):
        d = _seg_dist(Xs, Ys, x1, y1, x2, y2)
        metrics[f"fault_{i}_dist"] = d
        if d < cfg.fault_radius_frac:
            triggers.append((3, RiskLevel.RED,
                             f"羽流跨越断层 #{i}（距 {d:.3f} < {cfg.fault_radius_frac}），"
                             f"断层活化泄漏高风险（机制3）。"))
        elif d < cfg.fault_radius_frac + cfg.yellow_proximity_frac:
            triggers.append((3, RiskLevel.YELLOW,
                             f"羽流接近断层 #{i}（距 {d:.3f}），关注断层活化（机制3预警）。"))

    # ---- UQ 驱动预警 ----
    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=np.float32)
        mean_unc = float(unc.mean())
        metrics["uncertainty_mean"] = mean_unc
        uband = ((field > cfg.sat_threshold) |
                 (field + cfg.uncertainty_k * unc > cfg.sat_threshold))
        if uband.any():
            uy, ux = np.where(uband)
            UYs = (uy + 0.5) / H
            UXs = (ux + 0.5) / W
            for i, (wx, wy) in enumerate(cfg.wells):
                if _point_dist(UXs, UYs, wx, wy) < cfg.well_radius_frac:
                    triggers.append((2, RiskLevel.RED,
                                     f"不确定性带触及废弃井筒 #{i}，即便点预测未触也列入红警（机制2）。"))
            for i, ((x1, y1), (x2, y2)) in enumerate(cfg.faults):
                if _seg_dist(UXs, UYs, x1, y1, x2, y2) < cfg.fault_radius_frac:
                    triggers.append((3, RiskLevel.RED,
                                     f"不确定性带跨越断层 #{i}，列入红警（机制3）。"))
        if mean_unc > cfg.uncertainty_threshold:
            triggers.append((-1, RiskLevel.YELLOW,
                             f"模型不确定性偏高（均值 {mean_unc:.3f} > {cfg.uncertainty_threshold}），"
                             f"预测可信度不足，列入关注。"))

    # ---- 定级（取所有触发的最高级，显式、无文本解析） ----
    if not triggers:
        if uncertainty is not None and metrics.get("uncertainty_mean", 0.0) > cfg.uncertainty_threshold:
            level = RiskLevel.YELLOW
            reasons = [f"无显著羽流，但模型不确定性偏高，列入关注。"]
        else:
            level = RiskLevel.GREEN
            reasons = ["羽流被良好圈闭，远离盖层/边界/井筒/断层，且不确定性低，安全。"]
        mechs: List[int] = []
    else:
        level = max(t[1] for t in triggers)
        mechs = sorted({t[0] for t in triggers if t[0] > 0})
        reasons = [t[2] for t in triggers]

    return RiskResult(level, mechs, reasons, metrics)


# --------------------------------------------------------------------------- #
# 站点指标 → 跨样本相对分级（新口径）
# --------------------------------------------------------------------------- #
# 为什么需要这一套（本机实测，2026-10-30，官方权重 + 500 测试样本）：
#   ① 储层掩膜就是网格的前 52/96 行、顶行 = 0 → "前锋距顶 0 行"等于"到了模型上边界"，
#      100 样本 × 24 帧里 69.1% 的帧 gap=0、97.0% 的样本至少一帧顶到上界，
#      绝对几何带阈值在这里没有区分力（这就是旧的 assess() 在本数据上判全红的原因）。
#   ② 真值场的"掩膜外信号占比"恒为 0，而预测场 p50=0.114 / p90=0.623 → 它量的是
#      **模型在无监督区域的伪影**，属"可信度"轴，绝不能当泄漏证据。
#   ③ 真正跨样本有差异、且预测与真值高度一致的量（100 样本上 corr(pred,y)）：
#      羽流面积峰值 0.998、增长斜率 0.996、突破时间 0.941、侧向越界占比 0.932。
#   → 所以分级 = 少数几个"排序稳定"的指标，阈值锚在**数值真值**的分位数上，
#     再用保形区间把"模型说不准"的场地降到黄。
D_Z_MAX_M, D_X_MAX_M = 200.0, 3500.0
T_OUT = 24


def physical_time_days(n: int = T_OUT) -> np.ndarray:
    """24 个输出时刻对应的注入天数（官方 notebook 同款公式），末帧 ≈ 30 年。"""
    return np.cumsum(np.power(1.421245, np.arange(n)))


@dataclass
class SiteIndicators:
    """一个场地（一个样本）的可排序指标；全部由预测场 + 掩膜算出，不引入先验阈值。"""
    breakthrough_days: float      # 羽流首次到达储层顶行的时间；从未 → nan（最安全）
    ratio_peak: float             # 末帧羽流占储层面积比（本数据里等于峰值）
    ratio_mean: float             # 24 帧平均占比（累计充填程度）
    growth_slope: float           # 末帧 − 中位帧，仍在长大的幅度
    lateral_share: float          # 触及模型左右边界列的羽流格点占比
    outside_share: float          # 掩膜外信号占比（→ 可信度轴，不是风险轴）
    violation_share: float        # 饱和度越出 [0,1] 的格点占比（数值噪声看这个不够，见 violation_l1）
    violation_l1: float           # 越界的平均幅度（真值 ~2e-5 级，预测常在 1e-3 级）

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


def extract_indicators(sg: np.ndarray, mask_2d: np.ndarray, sat_thr: float = 0.01,
                       edge_cols: int = 5, days: Optional[np.ndarray] = None) -> SiteIndicators:
    sg = np.asarray(sg, dtype=np.float32)
    mask_2d = np.asarray(mask_2d, dtype=bool)
    if sg.ndim != 3:
        raise ValueError(f"需要 (H,W,T) 的饱和度立方体，实得 {sg.shape}")
    n_t = sg.shape[2]
    days = physical_time_days(n_t) if days is None else days
    if not mask_2d.any():
        return SiteIndicators(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    active_rows = np.where(mask_2d.any(axis=1))[0]
    top_row = int(active_rows[0])
    denom = max(int(mask_2d.sum()), 1)
    plume = (np.abs(sg) > sat_thr)
    inside = mask_2d[:, :, None]

    ratios = (plume & inside).sum(axis=(0, 1)) / denom
    # 前锋到达储层顶行 = 顶行上有超过阈值的 masked 格点
    top_hit = plume[top_row, :, :] & mask_2d[top_row, :, None]
    reaches = np.where(top_hit.any(axis=0))[0]
    bt = float(days[reaches[0]]) if reaches.size else float("nan")

    outside = float((plume & ~inside).sum()) / max(int(plume.sum()), 1)
    edge = np.zeros(mask_2d.shape, dtype=bool)
    if edge_cols > 0:
        edge[:, :edge_cols] = True
        edge[:, -edge_cols:] = True
    lateral = ((plume & edge[:, :, None]).sum()) / max(int(plume.sum()), 1)
    viol = ((sg < 0.0) | (sg > 1.0)) & inside
    # 越界幅度：负值主要来自"真值恰好为 0 的干格点上模型给出 -1e-3 级噪声"，
    # 所以占比（share）会虚高到几十个百分点，幅度（l1）才是能判断严重性的量。
    amount = np.maximum(sg - 1.0, 0.0) + np.maximum(-sg, 0.0)
    inside3 = np.broadcast_to(inside, sg.shape)  # inside 是 (H,W,1)，做布尔索引必须先铺满时间轴
    return SiteIndicators(bt, float(ratios[-1]), float(ratios.mean()),
                          float(ratios[-1] - np.median(ratios)), float(lateral),
                          float(outside), float(viol.sum() / max(denom * n_t, 1)),
                          float(amount[inside3].mean()))


@dataclass
class TierPolicy:
    """分级策略。阈值是"真值队列分位数"而非绝对量 —— 改数据集必须重算 RiskReference。"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "breakthrough": 0.45,   # 多早充满储层上部：封存项目最直接的时间判据
        "areal": 0.35,          # 羽流规模
        "lateral": 0.20,        # 侧向铺展到模型边界
    })
    red_pct: float = 0.90       # 综合分在**真值队列**里的分位 ≥ 此 → 红（队列最差的 10%）
    yellow_pct: float = 0.75    # ≥ 此 → 黄
    low_conf_pct: float = 0.90  # 掩膜外伪影/物理违背处于队列前 10% → 可信度不足

    def __post_init__(self):
        if not 0 < self.yellow_pct < self.red_pct < 1:
            raise ValueError("需要 0 < yellow_pct < red_pct < 1")
        s = sum(self.weights.values())
        if s <= 0:
            raise ValueError("weights 全为 0，无法综合")
        self.weights = {k: v / s for k, v in self.weights.items()}


class RiskReference:
    """由**真值场**（或指定的参照场）指标构成的经验分布，分位数的参照系。"""
    AXES = {  # key: (指标字段名, 数值越大风险越高?)
        "breakthrough": ("breakthrough_days", False),
        "areal": ("ratio_peak", True),
        "lateral": ("lateral_share", True),
    }

    def __init__(self, inds: List[SiteIndicators]):
        self.n = len(inds)
        if self.n == 0:
            raise ValueError("参照队列为空")
        self.inds = list(inds)
        self.ref = {k: np.sort(np.array([getattr(i, f) for i in inds], dtype=np.float64))
                    for k, (f, _) in self.AXES.items()}
        self._rank = {name: np.sort(np.array([getattr(i, name) for i in inds], dtype=np.float64))
                      for name in ("outside_share", "violation_l1")}
        self._score_cohort_cache: Dict[tuple, np.ndarray] = {}

    def axis_percentile(self, key: str, value: float) -> float:
        """该指标值在参照队列中的分位（0–1，越大越差）。突破时间是"越早越差"→ 取 1−分位。

        用 side="left"（队列里严格小于它的占比）而不是 "right"：
        队列退化（所有值相同，例如真值的掩膜外占比恒为 0）时，"right" 会把人人算成 1.0、
        于是全员告警；"left" 给出 0.0，即"没人比队列更好或更差" → 不触发，符合本意。
        """
        field_name, higher_worse = self.AXES[key]
        arr = self.ref[key]
        if not np.isfinite(value):
            return 0.0            # 从未到达顶行 = 最安全；其余轴不该出现 NaN
        cdf = float(np.searchsorted(arr, value, side="left")) / arr.size
        return cdf if higher_worse else max(0.0, 1.0 - cdf)

    def confidence_percentile(self, name: str, value: float) -> float:
        """可信度轴分位。参照队列退化（全等，例如**真值**的掩膜外占比恒为 0）时返回 0：
        退化分布里"人人并列"，此时若还拿 side 计算就会把任何非零值算成 1.0，
        于是每个场地都被"可信度不足"降级 —— 这是实测踩过的坑，别再用真值队列当可信度参照。"""
        arr = self._rank[name]
        if arr[-1] <= arr[0]:
            return 0.0
        return float(np.searchsorted(arr, value, side="left")) / arr.size

    def score_cohort(self, policy: "TierPolicy") -> np.ndarray:
        """队列里每个场地在该策略下的综合分，升序。红线/黄线的参照系就是这条分布。"""
        key = (tuple(sorted(policy.weights.items())), policy.red_pct, policy.yellow_pct)
        if key not in self._score_cohort_cache:
            self._score_cohort_cache[key] = np.sort(np.array(
                [composite_score({k: self.axis_percentile(k, getattr(i, f))
                                 for k, (f, _) in self.AXES.items()}, policy)
                 for i in self.inds], dtype=np.float64))
        return self._score_cohort_cache[key]

    def score_percentile(self, score: float, policy: "TierPolicy") -> float:
        """把"三个轴分位的加权和"换成它在队列里的分位。

        必须换，不能直接拿加权和去比 red_pct：加权和并不是均匀分布在 [0,1]，
        它挤在中间。实测 100 个场地（真值场）最大只到 0.890 —— 按绝对值比 0.90，
        红线永远无人可越，三级就退化成两级。根因是各轴高度相关（Spearman 0.94~0.998）
        且突破时间严重并列（97/100 场地在第一个输出时刻 1 天就顶到上界），
        所以 0.45 的突破权重几乎人人同样拿到，压不出尾部。"""
        arr = self.score_cohort(policy)
        if arr[-1] <= arr[0]:
            return 0.0
        return float(np.searchsorted(arr, score, side="left")) / arr.size


def composite_score(axes: Dict[str, float], policy: TierPolicy) -> float:
    return float(sum(policy.weights[k] * v for k, v in axes.items()))


def relative_tier(score: float, policy: TierPolicy) -> RiskLevel:
    if score >= policy.red_pct:
        return RiskLevel.RED
    if score >= policy.yellow_pct:
        return RiskLevel.YELLOW
    return RiskLevel.GREEN


def interval_tier(lo: float, hi: float, policy: TierPolicy) -> RiskLevel:
    """区间判级：只有"最悲观也不到黄线"才给绿，"最乐观也过红线"才给红，中间一律黄。"""
    if lo >= policy.red_pct:
        return RiskLevel.RED
    if hi < policy.yellow_pct:
        return RiskLevel.GREEN
    return RiskLevel.YELLOW


@dataclass
class SiteClassification:
    level: RiskLevel
    score: float
    score_lo: float
    score_hi: float
    axes: Dict[str, float]
    reasons: List[str]

    def __str__(self) -> str:
        ax = " ".join(f"{k}={v:.2f}" for k, v in self.axes.items())
        return (f"[{self.level.label}] 综合分位 {self.score:.3f}"
                f"（区间 {self.score_lo:.3f}–{self.score_hi:.3f}）| {ax}\n"
                + "\n".join(f"  - {r}" for r in self.reasons))


def classify_site(ind: SiteIndicators, ref: RiskReference, policy: TierPolicy,
                  q: Optional[Dict[str, float]] = None,
                  conf_ref: Optional[RiskReference] = None) -> SiteClassification:
    """点判级 + 区间判级 + 可信度降级。q 为各指标的保形半径（缺省=无 UQ，只按点判）。

    conf_ref 默认同 ref；但本数据集里**真值**的掩膜外占比恒为 0（退化分布，人人并列），
    拿它当参照等于永远不触发降级 —— 所以可信度轴要用**预测队列**建参照，由调用方传入。
    """
    axes = {k: ref.axis_percentile(k, getattr(ind, f)) for k, (f, _) in ref.AXES.items()}
    raw = composite_score(axes, policy)

    if q is None:
        raw_lo = raw_hi = raw
        axes_lo = axes_hi = axes
    else:
        axes_lo, axes_hi = {}, {}
        for k, (field_name, higher_worse) in ref.AXES.items():
            v = getattr(ind, field_name)
            if not np.isfinite(v):
                axes_lo[k] = axes_hi[k] = 0.0
                continue
            d = float(q.get(k, 0.0))
            v_low_risk, v_high_risk = (v - d, v + d) if higher_worse else (v + d, v - d)
            axes_lo[k] = ref.axis_percentile(k, v_low_risk)
            axes_hi[k] = ref.axis_percentile(k, v_high_risk)
        raw_lo = composite_score(axes_lo, policy)
        raw_hi = composite_score(axes_hi, policy)

    # 阈值比的是"队列分位"，不是加权和本身（见 score_percentile 的实测理由）
    score = ref.score_percentile(raw, policy)
    lo = ref.score_percentile(raw_lo, policy)
    hi = ref.score_percentile(raw_hi, policy)

    level = interval_tier(lo, hi, policy) if q is not None else relative_tier(score, policy)
    reasons: List[str] = []
    if q is not None:
        if lo >= policy.red_pct:
            reasons.append(f"即使取最乐观区间（队列分位下界 {lo:.2f}）仍在红线（{policy.red_pct:.2f}）之上 → 红。")
        elif hi < policy.yellow_pct:
            reasons.append(f"即使取最悲观区间（队列分位上界 {hi:.2f}）也不到黄线（{policy.yellow_pct:.2f}）→ 绿。")
        else:
            reasons.append(f"保形区间 [{lo:.2f},{hi:.2f}] 跨越阈值线 → 判黄："
                           "黄档的含义是『模型说不准』，不是『风险中等』。")
    else:
        reasons.append(f"无 UQ 区间，按队列分位 {score:.2f} 判级。")

    cref = conf_ref or ref
    conf = {k: cref.confidence_percentile(k, getattr(ind, k)) for k in cref._rank}
    low = [k for k, v in conf.items() if v >= policy.low_conf_pct]
    if low and level == RiskLevel.GREEN:
        level = RiskLevel.YELLOW
        reasons.append(f"可信度不足（{'、'.join(low)} 处于队列前 {1-policy.low_conf_pct:.0%}），不得判绿。")
    elif low:
        reasons.append(f"注意：{'、'.join(low)} 偏高，属模型在无监督区域的伪影，需人工复核，不代表泄漏。")
    return SiteClassification(level, score, lo, hi, axes, reasons)


# --------------------------------------------------------------------------- #
# 自检：验证逻辑会正确触发三档与 6 机制（不验证"报得对"）
# --------------------------------------------------------------------------- #
def _selftest() -> bool:
    H = W = 48
    # 场景 A：良好圈闭的小羽流（居中）→ 绿
    pa = np.zeros((H, W), np.float32)
    pa[24:30, 22:28] = 0.3
    ra = assess(pa, config=RiskConfig())
    assert ra.level == RiskLevel.GREEN, f"A 应绿，得 {ra.level}"

    # 场景 B：羽流触及废弃井 → 红（机制2）
    pb = np.zeros((H, W), np.float32)
    pb[24:30, 10:14] = 0.3
    cfg_b = RiskConfig(wells=[(0.25, 0.55)])
    rb = assess(pb, config=cfg_b)
    assert rb.level == RiskLevel.RED and 2 in rb.mechanism_ids, f"B 应红机制2，得 {rb}"

    # 场景 C：高不确定性但几何安全 → 黄
    pc = np.zeros((H, W), np.float32)
    pc[24:30, 22:28] = 0.3
    uc = np.full((H, W), 0.2, np.float32)
    rc = assess(pc, uncertainty=uc, config=RiskConfig())
    assert rc.level == RiskLevel.YELLOW, f"C 应黄，得 {rc.level}"

    # 场景 D：羽流突破顶部 → 红（机制1 或 6）
    pd = np.zeros((H, W), np.float32)
    pd[0:3, 20:28] = 0.3
    rd = assess(pd, config=RiskConfig())
    assert rd.level == RiskLevel.RED and (1 in rd.mechanism_ids or 6 in rd.mechanism_ids), f"D 应红机制1/6，得 {rd}"

    # 场景 E：羽流接近边界（预警）→ 黄（机制4）
    pe = np.zeros((H, W), np.float32)
    pe[20:26, 3:6] = 0.3  # x 中心约 0.09，处于 yellow 容差内、未进红区
    re = assess(pe, config=RiskConfig())
    assert re.level == RiskLevel.YELLOW and 4 in re.mechanism_ids, f"E 应黄机制4，得 {re}"

    # 场景 F：羽流跨越断层 → 红（机制3）
    pf = np.zeros((H, W), np.float32)
    pf[20:30, 23:27] = 0.3
    cfg_f = RiskConfig(faults=[((0.5, 0.0), (0.5, 1.0))])  # 竖穿中部的断层
    rf = assess(pf, config=cfg_f)
    assert rf.level == RiskLevel.RED and 3 in rf.mechanism_ids, f"F 应红机制3，得 {rf}"

    # 场景 G：指标提取 —— 顶行何时被羽流到达 / 掩膜外伪影 / 物理违背
    Hg, Wg, Tg = 24, 30, 8
    m_g = np.zeros((Hg, Wg), bool)
    m_g[:12, :] = True                      # 储层 = 前 12 行（顶行 0），与本数据集同构
    sg_g = np.zeros((Hg, Wg, Tg), np.float32)
    sg_g[6:12, 10:20, :] = 0.4              # 全程只在中间，永不顶到行 0
    sg_g[0, 14:16, 5:] = 0.5                # 从第 5 帧起到达顶行
    sg_g[20, 3:5, 2:] = 0.5                 # 掩膜外的信号（伪影）
    sg_g[3, 3, 6] = 1.7                     # 饱和度 >1，物理违背
    days_g = physical_time_days(Tg)
    ind_g = extract_indicators(sg_g, m_g, days=days_g)
    assert ind_g.breakthrough_days == float(days_g[5]), f"突破时间应={days_g[5]:.0f}，实得 {ind_g.breakthrough_days}"
    assert ind_g.outside_share > 0, "掩膜外信号没被算出来"
    assert ind_g.violation_share > 0, "饱和度越界没被算出来"
    assert ind_g.ratio_peak > 0, "面积比应为正"
    sg_nope = np.zeros((Hg, Wg, Tg), np.float32)
    sg_nope[6:12, 10:20, :] = 0.4            # 中间有羽流但永不顶到行 0
    ind_nope = extract_indicators(sg_nope, m_g, days=days_g)
    assert np.isnan(ind_nope.breakthrough_days), "未到达顶行时突破时间应为 NaN"
    assert ind_nope.outside_share == 0.0, "掩膜内干净场不该有越界信号"

    # 场景 H：相对分级 —— 参照系来自真值队列，红线只给队列最差的 10%
    cohort = [SiteIndicators(float(d), 0.2 + 0.02 * i, 0.1, 0.05, 0.1, 0.0, 0.0, 0.0)
              for i, d in enumerate(range(10, 250, 10))]      # 24 个场地，突破从 10d 到 240d
    ref_g = RiskReference(cohort)
    pol = TierPolicy()
    worst = SiteIndicators(1.0, 0.9, 0.5, 0.4, 0.9, 0.0, 0.0, 0.0)
    best = SiteIndicators(float("nan"), 0.01, 0.005, 0.0, 0.001, 0.0, 0.0, 0.0)
    cw = classify_site(worst, ref_g, pol)
    cb = classify_site(best, ref_g, pol)
    assert cw.level == RiskLevel.RED and cw.score >= pol.red_pct, f"最差场地应红，得 {cw.level} score={cw.score:.3f}"
    assert cb.level == RiskLevel.GREEN and cb.score < pol.yellow_pct, f"最好场地应绿，得 {cb.level}"
    # 单调性：面积比升高，综合分位不降
    s_prev = None
    for ratio in (0.1, 0.2, 0.3, 0.5, 0.9):
        axes = {"breakthrough": ref_g.axis_percentile("breakthrough", 100.0),
                "areal": ref_g.axis_percentile("areal", ratio),
                "lateral": ref_g.axis_percentile("lateral", 0.1)}
        s = composite_score(axes, pol)
        assert s_prev is None or s >= s_prev - 1e-12, "面积比越大分位应单调不降"
        s_prev = s

    # 场景 I：区间判级 —— 只有整条区间都在阈值同侧才给确定档
    assert interval_tier(0.92, 0.99, pol) == RiskLevel.RED, "下界已过红线应判红"
    assert interval_tier(0.30, 0.70, pol) == RiskLevel.GREEN, "上界未到黄线应判绿"
    assert interval_tier(0.60, 0.95, pol) == RiskLevel.YELLOW, "跨越阈值线必须降到黄"
    assert interval_tier(0.76, 0.95, pol) == RiskLevel.YELLOW, "点判红但下界不足红线 → 黄（说不准）"
    # 点判红 + 区间宽 → 自动降级为黄，这是 UQ 真正改变结论的例子
    wide = classify_site(worst, ref_g, pol, q={"breakthrough": 500.0, "areal": 0.5, "lateral": 0.5})
    assert wide.level == RiskLevel.YELLOW, f"大区间应把红降成黄，得 {wide.level}"
    tight = classify_site(worst, ref_g, pol, q={"breakthrough": 0.5, "areal": 0.001, "lateral": 0.001})
    assert tight.level == RiskLevel.RED, f"小区间应保住红，得 {tight.level}"

    # 场景 J：可信度降级 —— 几何上像绿，但掩膜外伪影处于队列最差 10% → 不得判绿
    conf_ref = RiskReference([SiteIndicators(100.0, 0.3, 0.1, 0.05, 0.1, o / 100.0, 0.0, 0.0) for o in range(24)])
    j = classify_site(SiteIndicators(float("nan"), 0.01, 0.005, 0.0, 0.001, 0.24, 0.0, 0.0), conf_ref, pol)
    assert j.level == RiskLevel.YELLOW, f"可信度不足应不得判绿，得 {j.level}"
    # 场景 K：退化参照（真值的掩膜外占比恒为 0）不该触发降级 —— 这是实测踩过的坑
    degen = RiskReference([SiteIndicators(100.0, 0.3, 0.1, 0.05, 0.1, 0.0, 0.0, 0.0) for _ in range(24)])
    k = classify_site(SiteIndicators(float("nan"), 0.01, 0.005, 0.0, 0.001, 0.24, 0.05, 0.01), degen, pol)
    assert k.level == RiskLevel.GREEN, f"退化参照把人人判成最差、进而全员降级，得 {k.level}"

    print("[risk_model.selftest] 通过：几何六机制(绿/红/黄/红/黄/红) + 指标提取 + 相对分级单调性 + "
          "区间三态 + 可信度降级。")
    return True


if __name__ == "__main__":
    _selftest()
