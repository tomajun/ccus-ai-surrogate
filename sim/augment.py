"""
augment.py —— 物理保持的数据增强（纯 numpy，无需 torch）
==============================================================
为什么需要它？
  代理模型的训练样本来自合成器 sim/synthetic_field.py。数量我们说了算，
  但默认采样比较"单调"（均匀随机）。本模块提供几种"不改物理规律"的
  增强手段，用来扩容 + 提升模型鲁棒性。

  比喻：就像给同一套考题做了几个"合法变体"——
    ① 镜像版（轴对称翻转）：题目左右反过来，考的知识点没变；
    ② 慢放回放版（时间插值）：在已有帧之间插一帧，过程更顺滑；
    ③ 复印噪点版（测量噪声）：给图像加一点点复印颗粒，模拟真实观测误差；
    ④ 混合版（MixUp，谨慎）：两张卷子按比例拼，但有"拼出来的题未必合理"的风险。
  卷子考的知识（物理规律）没变，但考生见过的花样多了 -> 泛化更好。

数据契约（与 make_dataset / dataset.py 对齐）：
  fields : (N, T, H, W) 或 (T, H, W)  CO2 饱和度场，取值 [0, 0.85]
  params : (N, 3)  [injection_rate, caprock_perm, thickness]
  risks  : (N,)    0/1/2

设计原则：
  - 增强只动 fields（有时也动 params，如 MixUp）；绝不凭空伪造物理上不
    合理的结构。
  - 轴对称翻转成立的前提：注入点在 grid 正中央（synthetic_field.py 里
    x0 = grid_n/2），左右对称。我们的合成器满足，所以 flip 严格合法。
  - 标签处理：flip/noise 不改变"末帧"的物理语义，risk 标签可保留；
    time_interp 改了末帧，保守起见建议保留原标签（下游若要精确可对末帧
    重算 risk_score）。
"""
import numpy as np

CLIP_LOW = 0.0
CLIP_HIGH = 0.85  # 残余+流动饱和度上限，与 synthetic_field.py 保持一致


def flip_axis(fields: np.ndarray) -> np.ndarray:
    """
    沿水平轴（最后一维）左右镜像。
    支持 (T, H, W) 单样本或 (N, T, H, W) 批量。
    因注入点在中央、域对称，此操作物理严格成立。
    """
    return np.flip(fields, axis=-1)


def time_interp(fields: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    在相邻时间步之间线性插值插入帧，把时序"加密"。
    factor=2：每对相邻帧间插 1 帧 -> T -> 2T-1。
    支持 3D(T,H,W) / 4D(N,T,H,W)。
    注意：会改变末帧，下游 risk 标签如需精确应按末帧重算。
    """
    if factor < 2:
        return fields
    if fields.ndim == 3:
        T = fields.shape[0]
        out = [fields[t] for t in range(T - 1)]
        mids = [0.5 * (fields[t] + fields[t + 1]) for t in range(T - 1)]
        seq = []
        for t in range(T - 1):
            seq.append(out[t])
            seq.append(mids[t])
        seq.append(fields[-1])
        return np.stack(seq, axis=0)
    elif fields.ndim == 4:
        T = fields.shape[1]
        out = [fields[:, t] for t in range(T - 1)]
        mids = [0.5 * (fields[:, t] + fields[:, t + 1]) for t in range(T - 1)]
        seq = []
        for t in range(T - 1):
            seq.append(out[t])
            seq.append(mids[t])
        seq.append(fields[:, -1])
        return np.stack(seq, axis=1)
    else:
        raise ValueError("fields must be 3D (T,H,W) or 4D (N,T,H,W)")


def add_noise(fields: np.ndarray, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    """
    加小幅高斯"测量噪声"，模拟地震/井数据的观测误差。
    加完 clip 回 [0, 0.85]，保证不越界、不破坏数据契约。
    sigma 建议 <= 0.02，过大会淹没真实信号。
    """
    noise = rng.normal(0.0, sigma, size=fields.shape).astype(fields.dtype)
    return np.clip(fields + noise, CLIP_LOW, CLIP_HIGH)


def mixup(fields_a: np.ndarray, fields_b: np.ndarray,
          rng: np.random.Generator, lam: float = None) -> np.ndarray:
    """
    MixUp：两样本按 lambda 线性混合。
    ⚠️ 物理风险：混合出的场未必满足真实流动规律，仅作可选实验，
       不建议作为默认/主用增强。默认 lambda ~ Beta(0.4,0.4)。
    """
    if lam is None:
        lam = float(rng.beta(0.4, 0.4))
    return lam * fields_a + (1.0 - lam) * fields_b


def augment_dataset(fields: np.ndarray, params: np.ndarray, risks: np.ndarray,
                    rng: np.random.Generator,
                    do_flip: bool = True, do_noise: bool = True,
                    sigma: float = 0.01, do_time_interp: bool = False):
    """
    对整批做默认增强（flip + noise），返回"原批 + 增强批"拼接后的新批。
    默认关闭 time_interp（会改变 T，需单独处理）；如需开启置 do_time_interp=True。

    返回 (fields_aug, params_aug, risks_aug)，可直接喂给 np.savez。
    """
    stacks_f = [fields]
    stacks_p = [params]
    stacks_r = [risks]

    if do_flip:
        stacks_f.append(flip_axis(fields))
        stacks_p.append(params.copy())
        stacks_r.append(risks.copy())

    if do_noise:
        stacks_f.append(add_noise(fields, rng, sigma))
        stacks_p.append(params.copy())
        stacks_r.append(risks.copy())

    fields_aug = np.concatenate(stacks_f, axis=0)
    params_aug = np.concatenate(stacks_p, axis=0)
    risks_aug = np.concatenate(stacks_r, axis=0)
    return fields_aug, params_aug, risks_aug


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    f = (np.random.rand(10, 48, 48).astype(np.float32)) * 0.8
    p = np.random.rand(10, 3).astype(np.float32)
    r = np.random.randint(0, 3, size=10).astype(np.int64)

    f_flip = flip_axis(f)
    assert f_flip.shape == f.shape, "flip shape mismatch"

    f_ti = time_interp(f, factor=2)
    assert f_ti.shape[0] == 2 * f.shape[0] - 1, "time_interp shape mismatch"

    f_noise = add_noise(f, rng, 0.01)
    assert f_noise.min() >= 0.0 and f_noise.max() <= 0.85, "noise clip mismatch"

    fa, pa, ra = augment_dataset(f, p, r, rng, do_flip=True, do_noise=True)
    assert fa.shape[0] == 3 * f.shape[0], "augment_dataset count mismatch"
    assert pa.shape[0] == fa.shape[0] and ra.shape[0] == fa.shape[0]

    print("augment self-test OK:")
    print("  flip        ", f_flip.shape)
    print("  time_interp ", f_ti.shape)
    print("  noise       ", f_noise.shape, "range [%.3f, %.3f]" % (f_noise.min(), f_noise.max()))
    print("  dataset aug ", fa.shape, "(original + flip + noise)")
