"""
kimberlina_csv_to_npz.py —— 把 EDX 下载的 Kimberlina CSV 结构转成统一 .npz 切片
================================================================================
Kimberlina CO2 Saturation Models 在 EDX 上是 ASCII CSV（每个 simDDD 文件夹含 33 个时间步的
sim-mDDD_co2sat_timestep.csv.zip）。本脚本把它们转成本仓库统一合同所需的 .npz：
    每个 sim -> 一个 .npz，键 'CO2_saturation' = (33, H, W) float32
然后交给 sim/verify_data_anchor.py kimberlina --src <输出目录> 汇入主数据。

重要（务必先读）：
- 真实 CSV 的精确布局以你首次下载后为准。本脚本支持两种 layout：
    matrix : 每个 CSV 是 H 行 x W 列的二维矩阵（最常见，默认）
    xyz    : 每个 CSV 是三列 (ix, iy, value) 的点表，脚本按最大索引重建网格
  若首次运行布局不符，调 --layout 或看报错信息微调。
- Kimberlina 论文为 CC-BY 开放获取（Alumbaugh et al. 2023, Geoscience Data Journal），
  数据按 CC-BY 署名即可复用，无需单独许可；本脚本仅做格式转换，不碰许可。

子命令：
  selftest                 用假 CSV 结构跑通 转换 + verify 整条路径（不碰真实数据，沙箱可跑）
  convert --src <dir>      把 <dir> 下 simDDD 文件夹里的 csv.zip 转成 .npz 到 --out
"""
import argparse
import io
import os
import re
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

GRID_N = 48


def _parse_csv(raw_bytes: bytes, layout: str):
    """把 CSV 字节解析成二维 (H, W) 数组。"""
    text = raw_bytes.decode("utf-8", errors="ignore")
    if layout == "xyz":
        data = np.loadtxt(io.StringIO(text), delimiter=",")
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"xyz 布局期望三列 (ix,iy,value)，但得到 shape={data.shape}")
        ix = data[:, 0].astype(int)
        iy = data[:, 1].astype(int)
        val = data[:, 2]
        nx = int(ix.max()) + 1
        ny = int(iy.max()) + 1
        grid = np.zeros((nx, ny), dtype=np.float32)
        grid[ix, iy] = val.astype(np.float32)
        return grid
    # matrix：H 行 x W 列
    arr = np.loadtxt(io.StringIO(text), delimiter=",")
    return arr.astype(np.float32)


def _find_co2sat_zips(sim_dir: str):
    """在 sim 文件夹里找 CO2 饱和度的 csv.zip（按时间步排序）。"""
    zips = [f for f in os.listdir(sim_dir) if f.lower().endswith(".csv.zip")]
    # 优先含 co2sat / co2 的文件名
    co2 = [f for f in zips if "co2sat" in f.lower() or "co2" in f.lower()]
    pool = co2 if co2 else zips
    if not pool:
        return []
    # 从文件名提取时间步数字用于排序
    def _step(name):
        nums = re.findall(r"(\d+)", name)
        return int(nums[-1]) if nums else 0
    pool.sort(key=_step)
    return pool


def _convert_sim(sim_dir: str, out_dir: str, layout: str):
    pool = _find_co2sat_zips(sim_dir)
    if not pool:
        return None
    frames = []
    for name in pool:
        with zipfile.ZipFile(os.path.join(sim_dir, name)) as z:
            # zip 内应只有一个 csv
            csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            raw = z.read(csv_name)
        frames.append(_parse_csv(raw, layout))
    stack = np.stack(frames, axis=0).astype(np.float32)  # (33, H, W)
    stack = np.clip(stack, 0.0, 1.0)
    os.makedirs(out_dir, exist_ok=True)
    sim_name = os.path.basename(os.path.normpath(sim_dir))
    out_path = os.path.join(out_dir, f"{sim_name}.npz")
    np.savez(out_path, CO2_saturation=stack)
    print(f"  {sim_name}: {len(frames)} 帧 -> {stack.shape} -> {out_path}")
    return out_path


def cmd_convert(src: str, out: str, layout: str):
    print(f"[convert] 扫描 {src} 下的 sim* 文件夹，转换 CO2 饱和 CSV -> {out}")
    count = 0
    for entry in sorted(os.listdir(src)):
        full = os.path.join(src, entry)
        if os.path.isdir(full) and entry.lower().startswith("sim"):
            res = _convert_sim(full, out, layout)
            if res:
                count += 1
    if count == 0:
        print("[convert] 没找到任何 sim*/ 文件夹或 co2sat csv.zip。确认 --src 指向含 simDDD 的目录。")
        return
    print(f"[convert] 完成 {count} 个 sim。下一步：\n"
          f"  python sim/verify_data_anchor.py kimberlina --src {out}")


def cmd_selftest():
    """造假的 Kimberlina CSV 结构 -> 转换 -> 跑 verify 整条路径。"""
    print("=== selftest：用假 CSV 结构验证转换+汇入整条代码路径（不碰真实数据）===")
    tmp = tempfile.mkdtemp(prefix="ccus_kim_csv_")
    src = os.path.join(tmp, "download")
    out = os.path.join(tmp, "slices")
    os.makedirs(src, exist_ok=True)

    T, H, W = 33, 60, 60
    for sim in ("sim001", "sim002"):
        sim_dir = os.path.join(src, sim)
        os.makedirs(sim_dir, exist_ok=True)
        for t in range(T):
            mat = np.zeros((H, W), dtype=np.float32)
            cy = cx = H // 2
            r = 5 + t * 0.6
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * r ** 2))
            mat = (blob * (0.3 + 0.02 * t)).astype(np.float32)
            buf = io.BytesIO()
            np.savetxt(buf, mat, delimiter=",", fmt="%.6f")
            zpath = os.path.join(sim_dir, f"sim-m{sim[3:]}_co2sat_{t:03d}y.csv.zip")
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr(f"sim-m{sim[3:]}_co2sat_{t:03d}y.csv", buf.getvalue().decode("utf-8"))

    cmd_convert(src, out, "matrix")

    # 交给 verify_data_anchor 的 kimberlina 处理（会自动切片/重采样到 48x48）
    from sim.verify_data_anchor import _run_kimberlina
    ok = _run_kimberlina(out, axis=1, out=os.path.join(tmp, "kimberlina_slices.npz"))
    assert ok, "selftest 失败：kimberlina 汇入返回 NO-GO"
    print("selftest 通过：假 CSV -> .npz -> 切片汇入 -> 统一合同，整条路径逻辑无误。\n"
          "真实数据下到后，只要 layout 对得上（默认 matrix），convert + verify 一条龙即可。\n"
          f"（临时文件在 {tmp}，可忽略）")


def main():
    ap = argparse.ArgumentParser(description="Kimberlina CSV -> npz 转换")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="用假数据跑通转换+汇入路径（默认，无参数即执行）")
    pc = sub.add_parser("convert", help="转换真实下载的 CSV 到 npz")
    pc.add_argument("--src", required=True, help="含 simDDD 文件夹的下载根目录")
    pc.add_argument("--out", default="data/kimberlina_slices_raw", help="输出 npz 目录")
    pc.add_argument("--layout", default="matrix", choices=["matrix", "xyz"], help="CSV 布局")
    args = ap.parse_args()
    cmd = getattr(args, "cmd", None) or "selftest"
    if cmd == "selftest":
        cmd_selftest()
    elif cmd == "convert":
        cmd_convert(args.src, args.out, args.layout)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
