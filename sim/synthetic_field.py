"""
synthetic_field.py —— 轻量级 CO2 羽流合成模拟器（教学/参赛用）
=============================================================
为什么存在这个文件？
  真实的 CO2 地质封存数值模拟要用 MRST(MATLAB) 或 PorePy(Python)，
  门槛高、跑得慢。为了让本科生"开箱即跑"、不碰任何涉密真实数据，
  这里提供一个**物理上合理、但被大幅简化的 2D 羽流模型**，
  用来批量生成 (注入参数 -> CO2 饱和度场序列) 的成对训练样本。

  比喻：这就像用"简化版物理"搭了一个迷你实验室——
  不是真刀真枪的工业级仿真，但能产出形状、规律都对得上的羽流数据，
  足够训练 AI 代理模型，也足够在答辩里讲清楚方法。
  论文里要明确写：这是"教学级代理数据"，高保真数据用 MRST/PorePy 替换即可。

模型假设（2D 垂直剖面，grid_n x grid_n）：
  - 注入点位于底部中央，CO2 因浮力向上"舌头状"迁移；
  - 累计注入体积 V ~ 注入速率 Q * 时间步 t；
  - 纵向扩展 ry 随 V 增长；到达盖层后横向铺展 rx 增长（重力舌）；
  - 盖层渗透性 k 越大，横向铺展越快、泄漏风险越高；
  - 每个场景加少量随机扰动，保证样本多样性。
"""
import numpy as np


def generate_plume(params: dict, grid_n: int = 48, n_steps: int = 10, seed: int = 0):
    """
    生成单口井/单场景的 CO2 羽流时空演化。

    参数
    ----
    params : dict
        injection_rate 相对注入速率 (0.5~2.0)
        caprock_perm    盖层渗透性   (0.1~1.0，越大越易泄漏)
        thickness       储层厚度(网格单位, 20~40)
    grid_n : 网格边长
    n_steps: 时间步数
    seed   : 随机种子（每场景不同 -> 数据多样）

    返回
    ----
    fields : np.ndarray, shape (n_steps, grid_n, grid_n), 取值 [0, 0.85]
    """
    rng = np.random.default_rng(seed)
    Q = float(params["injection_rate"])
    k = float(params["caprock_perm"])
    H = float(params["thickness"])

    x0 = grid_n / 2.0                 # 注入点水平位置（中央）
    y_bottom = grid_n - 1.0           # 底部
    xs = np.arange(grid_n)
    X, Y = np.meshgrid(xs, xs)        # Y=0 为顶部(盖层)，向下为深度

    fields = []
    for t in range(1, n_steps + 1):
        V = Q * t                                   # 累计注入体积（相对）
        rise = min(0.6 * V, grid_n - 3.0)           # 上浮距离：注入越强，越靠近盖层（演示更直观）
        y_c = y_bottom - rise                       # 羽流中心深度
        ry = 1.0 + 0.25 * V                         # 纵向半径
        rx = 1.0 + 0.18 * V * (0.6 + 0.8 * k)       # 横向半径（受盖层渗透性放大）

        dx = (rng.random() - 0.5) * 1.0              # 随机扰动
        dy = (rng.random() - 0.5) * 0.6
        sat = np.exp(-(((X - x0 + dx) ** 2) / (2 * rx ** 2)
                       + ((Y - y_c + dy) ** 2) / (2 * ry ** 2)))
        sat = np.clip(sat, 0.0, 0.85)               # 残余+流动饱和度上限
        fields.append(sat)
    return np.stack(fields, axis=0)


def risk_score(params: dict, field_final: np.ndarray, grid_n: int = 48):
    """
    启发式泄漏风险评分 -> 三档标签（低/中/高）。
    物理直觉：CO₂ 羽流横向铺展越广、整体覆盖越大、盖层渗透性越高、
    注入速率越强，泄漏风险越高。
    返回 (score, label)，label: 0=低 1=中 2=高
    """
    lateral = float((field_final > 0.2).mean(axis=1).max())   # 最宽深度行上的铺展比例
    spread = float((field_final > 0.2).mean())                # 整体铺展占比
    k = float(params["caprock_perm"])
    Q = float(params["injection_rate"])
    score = 0.45 * lateral + 0.25 * spread + 0.20 * k + 0.10 * Q
    if score > 0.42:
        label = 2
    elif score > 0.30:
        label = 1
    else:
        label = 0
    return float(score), int(label)


if __name__ == "__main__":
    p = dict(injection_rate=1.5, caprock_perm=0.8, thickness=30.0)
    f = generate_plume(p, grid_n=48, n_steps=10, seed=0)
    s, lab = risk_score(p, f[-1])
    print("field shape:", f.shape, "| value range:", f.min(), f.max())
    print("risk score:", round(s, 3), "| label:", lab)
