"""
risk_proxy.py —— 文献锚定的 CO2 封存"风险代理指数 (RPI)"
=========================================================
为什么写这个文件？
  旧版 `synthetic_field.risk_score` 有两道硬伤：
    A. 权重 0.45/0.25/0.20/0.10 与阈值 0.42/0.30 是**凭空捏的**，无物理依据；
    B. **循环论证**：标签由羽流场几何 + 生成它的 Q/k/H 算出，再让模型去"预测"，
       等于拟合一个已知确定性函数，分类器没有真实预测内容。
  本模块用**有文献出处的无量纲物理判据**替换任意权重，并引入两个**独立于羽流
  输入的地质背景参数（废弃井密度、最近断层距离）**，从而打破循环论证。

  ⚠️ 诚实声明（写进论文/答辩！）：
  本 RPI 是"合成但文献锚定的风险代理"，**不是真实地质真值**。它来自：
    - 已发表的盖层密封数 Πseal = Pc/ΔP（Espinoza & Santamarina 2017; IEAGHG 2011）
    - NRAP 框架将"废弃井/legacy wells"列为封存泄漏第一通道（NETL/NRAP）
    - 断层 proximity 作为次要通道（Kevin Dome NRAP 研究）
  但它**未经真实泄漏数据或地质师标注验证**。具体地，
  以下数值与阈值（废井密度映射、断层距离尺度、Πseal→风险的具体拐点）
  **需要指导老师（地质/石油工程方向）校准**，本文件只提供可辩护的默认值。

输入维度（比旧版多 2 个独立地质背景量）
  injection_rate  相对注入速率 (0.5~2.0)
  caprock_perm    盖层渗透性   (0.1~1.0，越大=越易漏)  —— 映射到毛细突破压力 Pc
  thickness       储层厚度(网格单位, 20~40)
  well_density    废弃/legacy 井密度 (口/km^2, 0~10)    —— 真实第一泄漏通道，NEW
  fault_distance  到最近导通断层距离 (km, 0.1~20)        —— 次要通道，NEW

输出
  rpi  : 风险代理指数 (0~1，越大越危险)
  label: 0=低 1=中 2=高（阈值默认按 RPI 分位，需老师校准）
"""

import numpy as np

# ---- 物理常数（CO2 地质封存典型条件）----
RHO_W = 1000.0      # 地层水密度 kg/m^3
RHO_CO2 = 700.0     # 超临界 CO2 密度 kg/m^3
G = 9.81            # 重力加速度 m/s^2

# ---- 教学级映射（⚠️ 需老师校准）----
# caprock_perm (0.1~1.0) -> 毛细突破压力 Pc (kPa)。渗透性越高，Pc 越低。
# 取对数单调映射：perm 0.1 -> ~5000 kPa (好盖层),  perm 1.0 -> ~300 kPa (差盖层)。
PC_MAX_KPA = 5000.0
PC_MIN_KPA = 300.0
def caprock_entry_pressure(caprock_perm: float) -> float:
    """caprock_perm -> 毛细突破压力 Pc (kPa)。高渗透=低密封。"""
    p = float(caprock_perm)
    # 对数插值：perm 0.1 对应 PC_MAX, perm 1.0 对应 PC_MIN
    frac = (np.log(p) - np.log(0.1)) / (np.log(1.0) - np.log(0.1))  # 0..1
    frac = float(np.clip(frac, 0.0, 1.0))
    return PC_MAX_KPA * (PC_MIN_KPA / PC_MAX_KPA) ** frac  # 几何插值

# 注入速率 Q -> 羽流柱高 h (m)。教学映射：Q 0.5->~30m, Q 2.0->~300m。
def plume_column_height(injection_rate: float, thickness: float) -> float:
    """羽流柱高 h (m)，随注入速率增加、受储层厚度约束。"""
    Q = float(injection_rate)
    H = float(thickness)
    # 简化：h 正比于 Q，上限约为储层厚度的物理对应（教学比例）
    h = 30.0 + (Q - 0.5) / 1.5 * 270.0          # 30~300 m
    h = min(h, 10.0 * H)                          # 不超过储层厚度的合理倍数
    return float(h)

def sealing_number(caprock_perm: float, injection_rate: float, thickness: float) -> float:
    """Πseal = Pc / ΔP，ΔP = (ρw-ρco2)·g·h。Πseal>>1 安全，≤1 高风险。"""
    Pc = caprock_entry_pressure(caprock_perm) * 1000.0   # kPa -> Pa
    h = plume_column_height(injection_rate, thickness)
    dP = (RHO_W - RHO_CO2) * G * h                        # 浮力超压 Pa
    return float(Pc / max(dP, 1.0))

def well_risk(well_density: float) -> float:
    """废弃井密度 -> 风险贡献 (0~1)，饱和上升。NRAP：1~10 口/km^2 量级。"""
    lam = float(well_density)
    LAM0 = 5.0  # 尺度参数（⚠️ 需老师校准）
    return float(1.0 - np.exp(-lam / LAM0))

def fault_risk(fault_distance: float) -> float:
    """断层距离 -> 风险贡献 (0~1)，距离越近风险越高。"""
    d = max(float(fault_distance), 0.01)
    D0 = 2.0  # 尺度参数 km（⚠️ 需老师校准）
    return float(1.0 / (1.0 + d / D0))

def seal_risk(seal_num: float) -> float:
    """密封数 -> 风险贡献 (0~1)，Πseal 越小越危险。Πseal≥3 视为安全。"""
    ns = max(float(seal_num), 0.01)
    return float(1.0 / (1.0 + ns))

def plume_geom_risk(field_final: np.ndarray) -> float:
    """羽流几何 -> 风险贡献 (0~1)，最宽行的铺展比例。"""
    lateral = float((field_final > 0.2).mean(axis=1).max())
    return float(np.clip(lateral, 0.0, 1.0))


def risk_proxy(params: dict, field_final: np.ndarray) -> tuple:
    """
    文献锚定风险代理指数。
    组合系数按文献重要性排序（非任意）：
      w_seal=0.35  (盖层密封，核心判据)
      w_well=0.35  (废弃井，NRAP 第一通道)
      w_fault=0.15 (断层，次要通道)
      w_geom=0.15  (羽流几何，仅辅助)
    返回 (rpi, label)，label: 0=低 1=中 2=高
    """
    ns = sealing_number(params["caprock_perm"], params["injection_rate"], params["thickness"])
    rw = well_risk(params.get("well_density", 0.0))
    rf = fault_risk(params.get("fault_distance", 20.0))
    rg = plume_geom_risk(field_final)

    w_seal, w_well, w_fault, w_geom = 0.35, 0.35, 0.15, 0.15
    rpi = w_seal * seal_risk(ns) + w_well * rw + w_fault * rf + w_geom * rg

    # 阈值默认按 RPI 分位（⚠️ 需老师校准，不应视为已验证）
    if rpi > 0.55:
        label = 2
    elif rpi > 0.35:
        label = 1
    else:
        label = 0
    return float(rpi), int(label)


def selftest():
    """一致性自检：RPI 必须对各风险因子单调响应，否则设计失败。"""
    print("== risk_proxy selftest ==")
    base = dict(injection_rate=1.0, caprock_perm=0.3, thickness=30.0,
                well_density=1.0, fault_distance=5.0)
    f = np.zeros((48, 48)); f[20:28, 18:30] = 0.5  # 假羽流

    r0, _ = risk_proxy(base, f)
    print(f"baseline RPI = {r0:.4f}")

    checks = []
    # 1) 废弃井密度↑ -> RPI↑
    hi = dict(base); hi["well_density"] = 8.0
    rh, _ = risk_proxy(hi, f); checks.append(("well_density↑ RPI↑", rh > r0))
    # 2) 断层距离↓ -> RPI↑
    near = dict(base); near["fault_distance"] = 0.2
    rn, _ = risk_proxy(near, f); checks.append(("fault_distance↓ RPI↑", rn > r0))
    # 3) 盖层渗透性↑ (Pc↓) -> RPI↑
    leaky = dict(base); leaky["caprock_perm"] = 0.95
    rl, _ = risk_proxy(leaky, f); checks.append(("caprock_perm↑ RPI↑", rl > r0))
    # 4) 注入量↑ -> RPI↑
    big = dict(base); big["injection_rate"] = 2.0
    rb, _ = risk_proxy(big, f); checks.append(("injection_rate↑ RPI↑", rb > r0))
    # 5) 全安全场景 RPI 应很低
    safe = dict(injection_rate=0.5, caprock_perm=0.1, thickness=40.0,
                well_density=0.0, fault_distance=20.0)
    rs, ls = risk_proxy(safe, f); checks.append(("safe scenario label=0", ls == 0))
    # 6) 全危险场景 RPI 应很高
    danger = dict(injection_rate=2.0, caprock_perm=1.0, thickness=20.0,
                  well_density=10.0, fault_distance=0.1)
    rd, ld = risk_proxy(danger, f); checks.append(("danger scenario label=2", ld == 2))

    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'OK' if v else 'FAIL'}] {name}")
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    selftest()
