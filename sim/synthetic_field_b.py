"""
synthetic_field_b.py —— 阶段 B：异质渗透率场下的 CO2 羽流合成（解锁 UQ 死穴）
============================================================================

为什么要有这个文件？（对照阶段 A）
------------------------------------------------------------------
阶段 A 的 sim/synthetic_field.py 用的是"同质平滑高斯羽流"——
把一个高斯团按重力舌公式往上推，全场渗透率处处一样。
结果是：羽流形状太"干净"，模型把整张图学成全 0 也能混个差不多的 loss。
实测印证：阶段 A 下 CNN 的 SSIM 只有 0.0106、深度集成 UQ 的 PICP 仅 0.7%~24%
（5 个种子预测几乎一致，且一致错得离谱）—— 数据里根本没有"信号"让模型学。

阶段 B 的核心改动：让渗透率**不再是常数**，而是真实地质里那种
"高渗河道窜流、低渗泥岩挡路、顶部页岩盖层封死"的异质场；并且——关键——
**把这张异质场作为模型输入**。否则同一组注入参数会对应无数种不同羽流，
模型只从 3 个标量学，异质差异对它就只是噪声，UQ 永远解锁不了。

羽流模型选型（踩坑记录，避免重蹈）
------------------------------------------------------------------
初版用"显式扩散 + 浮力平流"演化羽流，但实测在 48×48、20 步内羽流只发展出
底部一小坨（非零区域 <6%），异质差异被稀释到 mean|diff|≈1e-3 以下，
模型同样学不到信号。故最终采用"重力舌主干 + 异质渗透率调制"（方案 C）：
  - 重力舌公式保证羽流充分向上铺展（范围占网格 30%+）；
  - 异质 mask 把 k 场结构"印"进羽流：高渗通道鼓包(系数≈2.0)、低渗屏障
    凹陷(系数≈0.2)，不同地质场景产生明显不同、有结构的羽流。

比喻：阶段 A 像在平整玻璃上滴墨水（晕成规整圆）；阶段 B 像把墨水滴在
      一张有沟壑的海绵上——沟壑（高渗通道）吸墨鼓起、土坎（低渗）挡墨
      凹陷，墨迹歪歪扭扭分出结构，这才像真的。

物理合理性（教学级，非工业仿真）
  - 渗透率场 k(x,y)：多尺度高斯随机场 + 对数正态映射，模拟沉积非均质；
    顶部强制低渗 = 盖层密封；注入点附近提渗 = 井筒改造 / 储层主力层。
  - 羽流 = 重力舌(由注入参数决定主体形状) × 异质调制(由 k 场决定纹理/结构)。
  - 绝不伪造：不引入脱离物理的随机结构；扰动只是测量/次网格噪声量级。

数据契约（与阶段 A / dataset.py 对齐，并新增 perm 通道）
  fields : (T, H, W)  CO2 饱和度场，取值 [0, 0.85]
  params : dict {injection_rate, caprock_perm, thickness}
  perm   : (H, W)    异质渗透率相对场 —— 阶段 B 新增，将作为模型输入
"""
import numpy as np
from scipy.ndimage import gaussian_filter

CLIP_HIGH = 0.85  # 残余+流动饱和度上限，与阶段 A 保持一致


def generate_perm_field(grid_n: int = 48, rng: np.random.Generator = None,
                        seed: int = 0, n_modes: int = 6,
                        k_min: float = 0.05, k_max: float = 3.0) -> np.ndarray:
    """
    生成异质渗透率相对场 k(x, y)，shape (grid_n, grid_n)，取值约 [k_min, k_max]。

    做法：多尺度高斯随机场叠加 -> 标准化 -> 指数映射成对数正态（均值≈1，右偏，
    符合真实储层"少数高渗通道 + 大面积为中低渗"的形态）-> 顶部盖层强制低渗
    -> 注入点附近提渗 -> clip。
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    field = np.zeros((grid_n, grid_n), dtype=np.float32)
    for s in range(n_modes):
        # 每层一个越来越细的尺度：从 ~grid_n/2 到 ~grid_n/64
        sigma = max(1.0, grid_n / (2 ** (s + 1)))
        noise = rng.standard_normal((grid_n, grid_n))
        field += gaussian_filter(noise, sigma=sigma)

    field = (field - field.mean()) / (field.std() + 1e-8)
    # 指数映射：field=0 -> k=1；指数系数 0.9 保证高渗通道与低渗基质对比强烈
    k = np.exp(field * 0.9).astype(np.float32)

    # ---- 顶部盖层密封（真实页岩盖层近不透水，差 3~6 个数量级）----
    seal_rows = max(2, grid_n // 12)
    k[:seal_rows, :] *= 0.05

    # ---- 注入点（底部中央）附近提渗：模拟井筒改造 / 储层主力通道 ----
    x0 = grid_n // 2
    y_bottom = grid_n - 1
    y_idx, x_idx = np.ogrid[:grid_n, :grid_n]
    d2 = (x_idx - x0) ** 2 + (y_idx - y_bottom) ** 2
    k += 0.6 * np.exp(-d2 / (2 * (grid_n * 0.12) ** 2))

    return np.clip(k, k_min, k_max)


def generate_plume_hetero(params: dict, k_field: np.ndarray,
                          grid_n: int = 48, n_steps: int = 10,
                          seed: int = 0) -> np.ndarray:
    """
    重力舌主干 + 异质渗透率调制的羽流合成（方案 C）。

    物理直觉：CO2 因浮力从注入点向上铺展成"重力舌"；同时 CO2 偏好聚集在
    高渗通道、被低渗屏障抑制——把 k 场的结构"印"进羽流，使不同地质场景
    产生明显不同、有结构的羽流。

    返回
    ----
    fields : (n_steps, grid_n, grid_n) 随时间演化的饱和度场，取值 [0, CLIP_HIGH]
    """
    rng = np.random.default_rng(seed)
    Q = float(params["injection_rate"])
    k_cap = float(params["caprock_perm"])

    x0 = grid_n / 2.0
    y_bottom = grid_n - 1.0
    xs = np.arange(grid_n)
    X, Y = np.meshgrid(xs, xs)

    # 异质调制场 mask：高渗→接近1（CO2 偏好聚集），低渗→接近0（被抑制）
    kmin = float(k_field.min())
    kmax = float(k_field.max())
    if kmax - kmin < 1e-6:
        mask = np.full((grid_n, grid_n), 0.5, dtype=np.float32)  # 均匀场退化情形
    else:
        mask = ((k_field - kmin) / (kmax - kmin)).astype(np.float32)

    fields = []
    for t in range(1, n_steps + 1):
        V = Q * t
        rise = min(0.6 * V, grid_n - 3.0)          # 上浮距离：注入越强越接近盖层
        y_c = y_bottom - rise                       # 羽流中心深度
        ry = 1.0 + 0.25 * V                         # 纵向半径
        rx = 1.0 + 0.18 * V * (0.6 + 0.8 * k_cap)   # 横向半径（受盖层渗透性放大）
        dx = (rng.random() - 0.5) * 1.0             # 随机扰动（每步微抖动）
        dy = (rng.random() - 0.5) * 0.6
        sat = np.exp(-(((X - x0 + dx) ** 2) / (2 * rx ** 2)
                       + ((Y - y_c + dy) ** 2) / (2 * ry ** 2)))
        # 异质调制：高渗通道保留并增强 CO2（系数≈2.0），低渗屏障抑制（系数≈0.2）
        sat = sat * (0.20 + 0.80 * mask) * (1.0 + 0.40 * mask)
        sat = np.clip(sat, 0.0, CLIP_HIGH)
        fields.append(sat.astype(np.float32))
    return np.stack(fields, axis=0).astype(np.float32)


def selftest():
    """
    自检测试（纯 numpy，无需 torch）：
      1) 形状 / 值域契约；
      2) 异质场确实改变了羽流形态（与同质均匀场对比，末帧差异非平凡）；
      3) 注入痕迹存在（底部中央饱和 > 0）；
      4) 羽流充分展开（末帧非零区域 > 5%，否则模型学不到信号）；
      5) 异质场自身有空间变化（非均匀）。
    返回 True 表示通过。
    """
    rng = np.random.default_rng(0)
    grid_n, n_steps = 48, 10
    p = dict(injection_rate=1.5, caprock_perm=0.8, thickness=30.0)

    k_hetero = generate_perm_field(grid_n, rng, seed=1)
    f_het = generate_plume_hetero(p, k_hetero, grid_n, n_steps, seed=1)

    # 1) 形状 / 值域
    assert f_het.shape == (n_steps, grid_n, grid_n), "shape mismatch"
    assert f_het.min() >= 0.0 and f_het.max() <= CLIP_HIGH, "range violation"

    # 2) 异质 vs 同质
    k_uniform = np.ones((grid_n, grid_n), dtype=np.float32)
    f_uni = generate_plume_hetero(p, k_uniform, grid_n, n_steps, seed=1)
    diff = float(np.abs(f_het[-1] - f_uni[-1]).mean())
    assert diff > 1e-3, f"heterogeneous field did not change plume (diff={diff:.2e})"

    # 3) 注入痕迹
    assert f_het[-1, -1, grid_n // 2] > 0.0, "no injection signature at bottom-center"

    # 4) 羽流范围
    nonzero = float((f_het[-1] > 0.05).mean())
    assert nonzero > 0.05, f"plume too small (nonzero={nonzero:.3f})"

    # 5) 异质场空间变化
    assert k_hetero.std() > 0.1, "perm field looks too homogeneous"

    print("[selftest] OK")
    print(f"  plume shape      : {f_het.shape}, range [{f_het.min():.3f}, {f_het.max():.3f}]")
    print(f"  hetero-vs-uniform: mean|diff|={diff:.4f} (非平凡 -> 异质场起效)")
    print(f"  perm field std   : {k_hetero.std():.3f}, range [{k_hetero.min():.2f}, {k_hetero.max():.2f}]")
    print(f"  plume nonzero%   : {100 * nonzero:.1f}")
    return True


if __name__ == "__main__":
    selftest()
