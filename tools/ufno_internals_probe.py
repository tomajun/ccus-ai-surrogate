# 把 U-FNO 拆开看：逐层形状、逐模块参数量、被保留的傅里叶模式在哪。
# 只读推理，不训练；输出打印到终端，另存两张图到 results/ufno_internals/。
#
#   .venv/Scripts/python.exe tools/ufno_internals_probe.py \
#       --model_path 数据集/sg_UFNO_4500_trained.pt --data_dir 数据集
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "repro", "ufno"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402

import ufno as _ufno  # noqa: E402
from ufno import Net3d, SpectralConv3d  # noqa: E402
from ufno_reader import open_split  # noqa: E402


def module_param_table(model):
    """按顶层属性聚合参数量（复数权重按 1 个元素计，与 count_params 同口径）。"""
    rows = {}
    for name, p in model.named_parameters():
        head = name.split(".")[1] if name.startswith("conv1.") else name.split(".")[0]
        n = 1
        for s in p.shape:
            n *= s
        rows[head] = rows.get(head, 0) + n
    return rows


def trace_shapes(model, x):
    """在关键子模块上挂 hook，打印一次前向里每一站的张量形状。"""
    lines = []
    block = model.conv1

    def tag(label):
        def hook(_m, inp, out):
            i = inp[0] if isinstance(inp, tuple) else inp
            lines.append((label, tuple(i.shape), tuple(out.shape)))
        return hook

    handles = [(name, mod.register_forward_hook(tag(name)))
               for name, mod in block.named_children()]
    with torch.no_grad():
        model(x)
    for _, h in handles:
        h.remove()
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--outdir", default="results/ufno_internals")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)

    import __main__
    for name in dir(_ufno):
        if isinstance(getattr(_ufno, name), type):
            setattr(__main__, name, getattr(_ufno, name))

    fresh = Net3d(10, 10, 10, 36)
    print("== 结构参数量（Net3d(10,10,10,width=36)，与权重无关）==")
    print("count_params() =", fresh.count_params())
    table = module_param_table(fresh)
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:>12,}")

    sc = fresh.conv1.conv0
    print("\n== 一个 SpectralConv3d 的内部 ==")
    print("weights1.shape =", tuple(sc.weights1.shape), sc.weights1.dtype)
    print("scale =", float(sc.scale))
    un = fresh.conv1.unet3
    print("\n== 一个 U_net 的内部（unet3）==")
    for name, mod in un.named_children():
        print(f"  {name:<12}", type(mod).__name__)

    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.eval()
    same = torch.equal(model.state_dict()["conv1.fc0.weight"].cpu(),
                       fresh.conv1.fc0.weight.cpu())
    print("\n权重文件里的 fc0 与新实例是否同值（都用 seed=0 初始化）:", same)

    a, u = open_split(args.data_dir, "test")
    x = torch.from_numpy(np.array(a[0:1])).to(device)
    print("\n真实样本 0：输入", tuple(x.shape), "目标", tuple(np.array(u[0]).shape))
    mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)
    print("掩膜（第 0 通道 t=0 非零）占比 = %.4f" % mask.float().mean().item())

    print("\n== 一次前向的形状流水 ==")
    xpad = torch.nn.functional.pad(
        torch.nn.functional.pad(x, (0, 0, 0, 8, 0, 8), "replicate"),
        (0, 0, 0, 0, 0, 0, 0, 8), "constant", 0)
    print(f"  {'输入':<18} {tuple(x.shape)}")
    print(f"  {'Net3d 补齐后':<18} {tuple(xpad.shape)}")
    for label, i_shape, o_shape in trace_shapes(model, x):
        print(f"  {label:<18} {i_shape} -> {o_shape}")
    with torch.no_grad():
        pred = model(x)
    print("  输出", tuple(pred.shape), "取值范围 %.4f ~ %.4f" % (pred.min().item(), pred.max().item()))

    os.makedirs(args.outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k, v in table.items() if v > 1000]
    vals = [table[k] for k in keys]
    groups = {
        "SpectralConv3d x6\n(4 corner blocks each)": sum(v for k, v in table.items() if k.startswith("conv") and v > 100_000),
        "U_net x3\n(conv+skip+deconv)": sum(v for k, v in table.items() if k.startswith("unet")),
        "1x1 bypass w0..w5": sum(v for k, v in table.items() if k.startswith("w")),
        "fc0/fc1/fc2\n(lift + project)": table.get("fc0", 0) + table.get("fc1", 0) + table.get("fc2", 0),
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(groups)
    ax.bar(range(len(names)), [groups[n] / 1e6 for n in names], color="#1A5276")
    for i, n in enumerate(names):
        ax.text(i, groups[n] / 1e6 + 0.4, f"{groups[n]:,}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, max(groups.values()) / 1e6 * 1.18)
    ax.set_ylabel("parameters (millions; complex weight counted as 1)")
    ax.set_title(f"Where the {fresh.count_params():,} parameters live (Net3d, width=36)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "figA_param_breakdown.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("\n分组参数量：", groups)

    # 频率图：把真值按 Net3d 的补边规则补成 (104,208,32)，再 rfftn，
    # 逐格标出 SimpleBlock3d 每一层真正会乘权重的模式格子。
    def pad_like_net3d(field):
        p = np.pad(field, ((0, 8), (0, 0), (0, 0)), mode="constant")   # 竖向补 0，同 Net3d
        p = np.pad(p, ((0, 0), (0, 8), (0, 0)), mode="edge")            # 水平 replicate
        p = np.pad(p, ((0, 0), (0, 0), (0, 8)), mode="edge")            # 时间 replicate
        return p

    field = np.array(u[0]).astype(np.float64)                           # (96,200,24)
    F3 = np.fft.rfftn(pad_like_net3d(field))                            # (104,208,17)
    amp = np.log10(np.abs(F3) + 1e-12)
    m1 = m2 = m3 = 10
    keep = np.zeros(F3.shape, dtype=bool)
    keep[:m1, :m2, :m3] = keep[-m1:, :m2, :m3] = True
    keep[:m1, -m2:, :m3] = keep[-m1:, -m2:, :m3] = True
    power = np.abs(F3) ** 2
    frac = [100 * power[keep].sum() / power.sum()]
    for j in range(1, 8):
        f2 = np.fft.rfftn(pad_like_net3d(np.array(u[j]).astype(np.float64)))
        pw = np.abs(f2) ** 2
        frac.append(100 * pw[keep].sum() / pw.sum())
    print("被保留的模式格子 = %d / %d = %.2f%%；前 8 个样本能量占比 = %.1f%%（min %.1f / max %.1f）"
          % (keep.sum(), keep.size, 100 * keep.mean(), np.mean(frac), min(frac), max(frac)))

    sl = amp[:, :, 0]                       # (104 竖向, 208 水平)：imshow 里 x=列=水平，y=行=竖向
    ks = keep[:, :, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    im0 = axes[0].imshow(sl, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("log10 |rfftn| of the padded plume volume, time-mode index 0\n(horizontal 208 x vertical 104)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    for x0 in (0, 198):
        for y0 in (0, 94):
            axes[0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), 10, 10,
                                            fill=False, ec="#C4622D", lw=2))
    axes[1].imshow(ks.astype(float), origin="lower", aspect="auto", cmap="gray_r")
    axes[1].set_title("cells one SpectralConv3d multiplies\n4 corner blocks x 10 time-modes = 4000 of 367744")
    for ax in axes:
        ax.set_xlabel("horizontal frequency index (0..207)")
        ax.set_ylabel("vertical frequency index (0..103)")
    fig.tight_layout()
    p2 = os.path.join(args.outdir, "figB_fourier_modes.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"\n图已存 -> {p1}\n        {p2}")


if __name__ == "__main__":
    main()
