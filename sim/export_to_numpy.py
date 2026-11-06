"""
export_to_numpy.py —— 把高保真仿真数据转成仓库统一训练格式
================================================================
为什么有这个文件？
    README 第 7 节承诺过的高保真替换路径：本仓库默认用 Python 合成数据"开箱即跑"，
    但如果你想用真实物理仿真（MRST/MATLAB 或 PorePy/Python）产出的数据来训练，
    就用这个脚本把它们的输出转成本仓库统一的 .npz 格式。
    转换后，**train.py / train_risk.py / dataset.py 一行都不用改**，直接吃新数据。

统一格式（与 sim/make_dataset.py 完全一致，键名固定）：
    fields : (N, T, grid_n, grid_n)  CO2 饱和度时空序列，取值建议 [0,1]
    params : (N, 3)                  [injection_rate, caprock_perm, thickness]
    risks  : (N,)                    0/1/2 泄漏风险标签（由 risk_score 自动算）

支持的输入：
    A. MRST 导出的 .mat：需 scipy（见 requirements-hifi.txt）。
       约定变量名：饱和度场放 'S'/'sat'/'saturation'（shape 可为 (T,H,W) 或 (H,W,T)），
       参数放 'params'（长度 3 的向量）。若 .mat 里没有 params，用同目录 params.json 补充（见下）。
    B. PorePy / 任意 Python 导出的 .npy 或 .npz：
       .npy 直接是场 (T,H,W) 或 (H,W,T)；
       .npz 用键 'field'（同上）和可选的 'params'（长度 3）。
       同样可用同目录 params.json 补充参数。

params.json 两种写法（放在 --src 目录下）：
    1) 按文件名映射： {"case001.mat": [1.5, 0.8, 30.0], "case002.mat": [0.9, 0.3, 25.0], ...}
    2) 按文件排序的顺序列表： [[1.5,0.8,30.0], [0.9,0.3,25.0], ...]
    每个三元组顺序固定为 [injection_rate, caprock_perm, thickness]。

用法：
    # MRST .mat 目录 -> 统一 npz
    python sim/export_to_numpy.py --src data_hifi/mats --out data/synthetic_hifi.npz
    # PorePy .npy/.npz 目录
    python sim/export_to_numpy.py --src data_hifi/porepy --out data/synthetic_hifi.npz
    # 之后照常用：
    python -m src.train --model cnn --data data/synthetic_hifi.npz
    python -m src.train_risk --data data/synthetic_hifi.npz
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from sim.synthetic_field import risk_score


def _orient(field: np.ndarray) -> np.ndarray:
    """把 3D 场统一成 (T, H, W)。

    启发式：如果第一维是三个维度里最小的，就认为它是时间维 (T,H,W)；
    否则认为时间维在最后 (H,W,T) -> 转置成 (T,H,W)。
    这是经验判断，若你的数据不符合，请在本函数里显式指定。
    """
    field = np.asarray(field)
    if field.ndim != 3:
        raise ValueError(f"场必须是 3D (T,H,W) 或 (H,W,T)，但拿到 ndim={field.ndim}")
    a, b, c = field.shape
    if a <= min(b, c):
        return field.astype(np.float32)          # (T,H,W)
    return field.transpose(2, 0, 1).astype(np.float32)  # (H,W,T) -> (T,H,W)


def _resample(field: np.ndarray, grid_n: int) -> np.ndarray:
    """把 (T,H,W) 重采样到 (T,grid_n,grid_n)。需要 scipy。"""
    if field.shape[1] == grid_n and field.shape[2] == grid_n:
        return field
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise RuntimeError(
            "重采样到目标网格需要 scipy。请先 `pip install scipy`（或见 requirements-hifi.txt），"
            "或保证输入数据已是 %d x %d 网格。" % (grid_n, grid_n)
        )
    factor = (1.0, grid_n / field.shape[1], grid_n / field.shape[2])
    return zoom(field, factor, order=1).astype(np.float32)


def _read_mat(path: str):
    try:
        from scipy.io import loadmat
    except ImportError:
        raise RuntimeError("读取 .mat 需要 scipy，请先 `pip install scipy`（见 requirements-hifi.txt）。")
    m = loadmat(path)
    field = None
    for key in ("S", "sat", "saturation", "CO2_sat"):
        if key in m and isinstance(m[key], np.ndarray) and m[key].ndim == 3:
            field = m[key]
            break
    if field is None:  # 退而求其次：取第一个 3D 数组
        for v in m.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                field = v
                break
    if field is None:
        raise ValueError(f"{path} 里找不到 3D 饱和度场，请检查变量名。")
    params = None
    if "params" in m and np.asarray(m["params"]).size >= 3:
        params = np.asarray(m["params"]).reshape(-1)[:3].astype(np.float32)
    return _orient(field), params


def _read_npy(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        if "field" in d:
            field = d["field"]
        else:
            # 退而求其次：取第一个 3D 数组
            field = next(v for v in d.values() if isinstance(v, np.ndarray) and v.ndim == 3)
        params = np.asarray(d["params"]).reshape(-1)[:3].astype(np.float32) if "params" in d else None
    else:  # .npy
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"{path} 应为 3D 场 (T,H,W) 或 (H,W,T)，但 ndim={arr.ndim}")
        field, params = arr, None
    return _orient(field), params


def _load_sidecar_params(src: str, ordered_names: list):
    """从 --src 目录的 params.json 读取参数；支持 dict(按文件名) 或 list(按排序顺序)。"""
    pj = os.path.join(src, "params.json")
    if not os.path.exists(pj):
        return None
    with open(pj, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [np.array(data[n], dtype=np.float32)[:3] for n in ordered_names]
    if isinstance(data, list):
        return [np.array(row, dtype=np.float32)[:3] for row in data]
    raise ValueError("params.json 必须是 dict(文件名->[Q,k,H]) 或 list([[Q,k,H],...])。")


def main(src: str, out: str, grid_n: int = 48):
    if not os.path.isdir(src):
        raise NotADirectoryError(f"--src 必须是目录：{src}")

    files = sorted(
        f for f in os.listdir(src)
        if f.lower().endswith((".mat", ".npy", ".npz"))
    )
    if not files:
        raise FileNotFoundError(f"{src} 下没有找到 .mat/.npy/.npz 文件。")

    sidecar = _load_sidecar_params(src, files)

    fields, params_arr, risks = [], [], []
    for i, name in enumerate(files):
        path = os.path.join(src, name)
        if name.lower().endswith(".mat"):
            field, p = _read_mat(path)
        else:
            field, p = _read_npy(path)

        if p is None:
            if sidecar is None:
                raise ValueError(
                    f"{name} 内无 params，且未提供 params.json。"
                    "请在 --src 放 params.json（见文件头说明）。"
                )
            p = sidecar[i]

        field = _resample(field, grid_n)
        field = np.clip(field, 0.0, 1.0)            # 饱和度归一，与合成数据口径一致

        pdict = {
            "injection_rate": float(p[0]),
            "caprock_perm": float(p[1]),
            "thickness": float(p[2]),
        }
        _, lab = risk_score(pdict, field[-1], grid_n)

        fields.append(field)
        params_arr.append([pdict["injection_rate"], pdict["caprock_perm"], pdict["thickness"]])
        risks.append(lab)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(
        out,
        fields=np.stack(fields).astype(np.float32),
        params=np.array(params_arr, dtype=np.float32),
        risks=np.array(risks, dtype=np.int64),
    )
    counts = np.bincount(np.array(risks), minlength=3)
    print(f"saved -> {out}")
    print(f"samples={len(fields)}  field_shape={fields[0].shape}  grid_n={grid_n}")
    print(f"risk label distribution (low/mid/high): {counts.tolist()}")
    print("提示：之后用 `python -m src.train --data %s` 训练即可，其余脚本无需改动。" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MRST/PorePy 高保真数据 -> 仓库统一 npz 格式")
    ap.add_argument("--src", required=True, help="含 .mat/.npy/.npz 的目录")
    ap.add_argument("--out", default="data/synthetic_hifi.npz", help="输出 .npz 路径")
    ap.add_argument("--grid_n", type=int, default=48, help="目标网格边长（默认 48，需与模型一致）")
    args = ap.parse_args()
    main(args.src, args.out, args.grid_n)
