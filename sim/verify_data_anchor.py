"""
verify_data_anchor.py —— CCUS 数据锚点最低成本验证
================================================
在决定 All-in CCUS 之前，用最小代价确认"反转条件 1"（真实/高保真数据锚）是否可达。

三条子命令：
  selftest    用合成假数据跑通整条代码路径，证明本脚本没问题（不碰任何真实数据，沙箱可跑）
  kimberlina  验证 Kimberlina 3D 饱和场能下载/切片/汇入统一 .npz 合同（需你先下好 .npz）
  natcarb     验证 NATCARB 真实参数先验是否已接通 sample_params（不再是占位）

统一合同（与 sim/export_to_numpy.py、sim/make_dataset.py 一致）：
  fields : (N, T, 48, 48)  CO2 饱和度时空序列
  params : (N, 3)          [injection_rate, caprock_perm, thickness]
  risks  : (N,)            0/1/2

用法：
  python sim/verify_data_anchor.py selftest
  python sim/verify_data_anchor.py kimberlina --src <目录>
  python sim/verify_data_anchor.py natcarb
"""
import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

GRID_N = 48


def _resample(field: np.ndarray, grid_n: int = GRID_N) -> np.ndarray:
    """(T,H,W) -> (T,grid_n,grid_n)。优先 scipy zoom；没有就就近取整（仅自检用）。"""
    if field.shape[1] == grid_n and field.shape[2] == grid_n:
        return field.astype(np.float32)
    try:
        from scipy.ndimage import zoom
        factor = (1.0, grid_n / field.shape[1], grid_n / field.shape[2])
        return zoom(field, factor, order=1).astype(np.float32)
    except ImportError:
        # 退化：用整数倍裁剪/填充，仅用于 selftest 不报错
        h, w = field.shape[1], field.shape[2]
        out = np.zeros((field.shape[0], grid_n, grid_n), dtype=np.float32)
        m = min(h, grid_n)
        out[:, :m, :m] = field[:, :m, :m]
        return out


def _slice_to_2d(vol: np.ndarray, axis: int) -> np.ndarray:
    """从 4D (T,*,*,*) 体积中取某个轴的中位切片 -> (T,H,W)。"""
    if vol.ndim != 4:
        raise ValueError(f"期望 4D 体积 (T,*,*,*)，但拿到 ndim={vol.ndim}, shape={vol.shape}")
    ax = axis if axis >= 0 else vol.ndim + axis
    idx = vol.shape[ax] // 2
    slicer = [slice(None)] * 4
    slicer[ax] = idx
    return vol[tuple(slicer)].astype(np.float32)  # (T, H, W)


def _find_saturation(npz, key=None):
    """在 Kimberlina 的多个数组里挑出 CO2 饱和度场（3D 或 4D）。"""
    keys = list(npz.keys())
    if key is not None:
        if key not in keys:
            raise KeyError(f"指定键 {key} 不在文件中，实际键：{keys}")
        return npz[key], key
    # 自动猜：优先含 'sat'/'co2' 的键
    for k in keys:
        if "sat" in k.lower() or "co2" in k.lower():
            return npz[k], k
    # 退而求其次：第一个 3D 或 4D 浮点数组
    for k in keys:
        a = npz[k]
        if isinstance(a, np.ndarray) and a.ndim in (3, 4) and np.issubdtype(a.dtype, np.floating):
            return a, k
    raise ValueError(f"找不到 3D/4D 饱和度数组，文件键：{keys}")


def cmd_selftest():
    """用假数据跑通：造一个 (T,D,H,W) 的假 Kimberlina 体积 -> 切片 -> 转统一 npz -> 断言合同。"""
    print("=== selftest：用假数据验证整条代码路径（不碰真实数据）===")
    tmp = tempfile.mkdtemp(prefix="ccus_verify_")
    src_dir = os.path.join(tmp, "kimberlina_fake")
    os.makedirs(src_dir, exist_ok=True)

    # 假体积：T=33, D=64, H=64, W=64，中央一条高斯羽流，逐步变粗
    T, D, H, W = 33, 64, 64, 64
    xs = np.arange(W)
    arr = np.zeros((T, D, H, W), dtype=np.float32)
    for t in range(T):
        cx = cy = D // 2
        r = 4 + t * 0.5
        yy, xx = np.meshgrid(np.arange(D), np.arange(W), indexing="ij")
        blob = np.exp(-(((xx - W // 2) ** 2) + ((yy - cy) ** 2)) / (2 * r ** 2))
        for d in range(D):
            arr[t, d] = blob * (0.3 + 0.02 * t)
    np.savez(os.path.join(src_dir, "fake_CO2_saturation.npz"), CO2_saturation=arr)

    # 跑 kimberlina 处理逻辑
    ok = _run_kimberlina(src_dir, axis=1, out=os.path.join(tmp, "kimberlina_slices.npz"))
    assert ok, "selftest 失败：kimberlina 处理返回 NO-GO"
    print("selftest 通过：假数据成功切成统一合同，脚本逻辑无误。\n"
          "下一步：在真实数据上跑 `kimberlina --src <目录>`。\n"
          f"（临时文件在 {tmp}，可忽略）")


def _run_kimberlina(src: str, axis: int = 1, out: str = "data/kimberlina_slices.npz", key=None):
    from sim.export_to_numpy import main as export_main

    if not os.path.isdir(src):
        raise NotADirectoryError(f"--src 必须是目录：{src}")
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"{src} 下没有 .npz 文件")

    slice_dir = tempfile.mkdtemp(prefix="ccus_slices_")
    print(f"[kimberlina] 扫描 {len(files)} 个文件，逐个切片 -> (T,{GRID_N},{GRID_N})")
    wrote = 0
    for name in files:
        d = np.load(os.path.join(src, name))
        try:
            vol, used_key = _find_saturation(d, key)
        except (KeyError, ValueError) as e:
            print(f"  跳过 {name}: {e}")
            continue
        if vol.ndim == 4:
            field = _slice_to_2d(vol, axis)          # (T,H,W)
        elif vol.ndim == 3:
            # 直接是 (T,H,W) 或 (H,W,T)；交给 export 的 _orient 处理
            field = vol
        else:
            print(f"  跳过 {name}: 维度 {vol.ndim} 不支持")
            continue
        field = _resample(field, GRID_N)
        field = np.clip(field, 0.0, 1.0)
        np.savez(os.path.join(slice_dir, name), field=field)
        wrote += 1
        print(f"  {name}: 键={used_key} 原始{vol.shape} -> 切片{field.shape}")

    # 重要：Kimberlina 给的是饱和度场，但不按我们的 3 参数合同提供注入参数。
    # 真实项目必须从 Kimberlina 的情景元数据（注入速率等）里补齐 params；
    # 这里若没有 params.json，就生成"合成占位参数"让管线能跑通验证，并显式警示。
    pj = os.path.join(slice_dir, "params.json")
    if not os.path.exists(pj):
        import random
        rng2 = random.Random(0)
        params_list = [
            [rng2.uniform(0.5, 2.0), rng2.uniform(0.1, 1.0), rng2.uniform(20.0, 40.0)]
            for _ in range(wrote)
        ]
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(params_list, f)
        print("  [注意] 未提供 params.json：已写入【合成占位参数】仅用于跑通验证。\n"
              "    真实使用必须从 Kimberlina 情景元数据补齐 injection_rate/caprock_perm/thickness，\n"
              "    否则 surrogate 的'参数->场'监督信号是假的。")

    if wrote == 0:
        print("[kimberlina] NO-GO：没有任何文件切出有效场。检查键名/维度（可用 --key 指定）。")
        return False

    export_main(slice_dir, out, GRID_N)
    # 断言合同
    z = np.load(out)
    fs = z["fields"].shape
    assert z["fields"].ndim == 4 and fs[2] == GRID_N and fs[3] == GRID_N, f"合同不符：{fs}"
    print(f"[kimberlina] GO：产出 {out}，fields{fs}，已符合统一合同。\n"
          f"注意：这只是技术可行性通过。Kimberlina 署名许可未到前，不得写进申报书/训练成果。")
    return True


def cmd_kimberlina(src: str, axis: int, key: str, out: str):
    _run_kimberlina(src, axis, out, key)


def cmd_netcab():
    """确认 NATCARB 真实先验已接通 sample_params。"""
    from sim.make_dataset import load_priors, sample_params

    priors_path = os.path.join(ROOT, "data", "priors.json")
    priors = load_priors(priors_path)
    if priors is None:
        print("[natcarb] NO-GO：data/priors.json 未启用（enabled != true）或不存在。\n"
              "  动作：把真实区间填进 params，再把 \"enabled\": false 改为 true，然后重跑本命令。")
        return
    print("[natcarb] GO：data/priors.json 已启用，sample_params 将按真实分布采样。\n"
          "  采样 2000 组，验证分布确实偏离均匀：")
    rng = np.random.default_rng(0)
    samples = [sample_params(rng, priors) for _ in range(2000)]
    arr = np.array([[s["injection_rate"], s["caprock_perm"], s["thickness"]] for s in samples])
    names = ["injection_rate", "caprock_perm", "thickness"]
    for i, nm in enumerate(names):
        col = arr[:, i]
        p10, p50, p90 = np.percentile(col, [10, 50, 90])
        print(f"  {nm:14s} mean={col.mean():.3f} min={col.min():.3f} max={col.max():.3f} "
              f"P10={p10:.3f} P50={p50:.3f} P90={p90:.3f}")
    print("[natcarb] 若分布明显非均匀（如 caprock_perm 偏向低值，贴合真实近不透水页岩），"
          "则'参数取自真实地质分布'这句话真实可辩护。")


def main():
    ap = argparse.ArgumentParser(description="CCUS 数据锚点验证")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="用假数据跑通代码路径")
    pk = sub.add_parser("kimberlina", help="验证 Kimberlina .npz 切片汇入合同")
    pk.add_argument("--src", required=True, help="含 Kimberlina .npz 的目录")
    pk.add_argument("--axis", type=int, default=1, help="4D 体积里取哪个轴做切片（默认 1）")
    pk.add_argument("--key", default=None, help="指定饱和度数组的键名（不指定则自动猜）")
    pk.add_argument("--out", default="data/kimberlina_slices.npz")
    sub.add_parser("natcarb", help="验证 NATCARB 真实先验已接通")

    args = ap.parse_args()
    if args.cmd == "selftest":
        cmd_selftest()
    elif args.cmd == "kimberlina":
        cmd_kimberlina(args.src, args.axis, args.key, args.out)
    elif args.cmd == "natcarb":
        cmd_netcab()


if __name__ == "__main__":
    main()
