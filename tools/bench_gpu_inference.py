"""GPU 推理基准：在真实 test 样本上计时 U-FNO 前向，并报告显存峰值。

不需要预训练权重（用随机初始化的同结构模型计时即可），也支持 --model_path 用官方权重复测。

用法（仓库根）：
    python tools/bench_gpu_inference.py --data_dir 数据集
    python tools/bench_gpu_inference.py --data_dir 数据集 --device cpu --runs 1
"""
import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "repro", "ufno"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.join(ROOT, "数据集"))
    p.add_argument("--model_path", default=None, help="官方预训练权重；缺省则用随机初始化的同结构模型计时")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--runs", type=int, default=3, help="计时次数（首次热身不计）")
    p.add_argument("--width", type=int, default=36)
    return p.parse_args()


def pick_device(spec):
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[错误] 指定了 --device cuda，但 torch.cuda.is_available() 为 False（CUDA 版 torch 未装好？）")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = pick_device(args.device)

    print(f"torch {torch.__version__} | cuda build: {torch.version.cuda}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} | 算力 {torch.cuda.get_device_capability(0)}")
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"显存: 空闲 {free_b / 2**30:.2f} GiB / 共 {total_b / 2**30:.2f} GiB")
    else:
        print("[提示] 当前用的是 CPU 路径（CPU 实测约 9.0 s/样本）。")

    test_a, test_u = None, None
    if os.path.exists(os.path.join(args.data_dir, "sg_test_a.pt")):
        from ufno_reader import open_split
        test_a, test_u = open_split(args.data_dir, "test")
        print(f"样本: test_a {tuple(test_a.shape)} | test_u {tuple(test_u.shape)}")
        x_np = np.array(test_a[0:1])
    else:
        print(f"[提示] 未找到 {args.data_dir}/sg_test_a.pt，改用随机张量计时。")
        x_np = np.random.randn(1, 96, 200, 24, 12).astype(np.float32)

    if args.model_path:
        model = torch.load(args.model_path, weights_only=False, map_location="cpu")
        print(f"加载权重: {args.model_path}")
    else:
        from ufno import Net3d
        model = Net3d(10, 10, 10, args.width)
        print(f"随机初始化 Net3d(10,10,10,width={args.width})（仅为计时，不代表精度）")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.3f} M")

    model.to(device).eval()
    x = torch.from_numpy(x_np).to(device)
    print(f"输入显存占用: {x.numel() * 4 / 2**20:.1f} MiB")

    torch.cuda.reset_peak_memory_stats(0) if device.type == "cuda" else None

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(0)
    warm = time.perf_counter() - t0
    print(f"热身(含首次 kernel 编译): {warm:.2f} s | 输出 {tuple(out.shape)}")

    times = []
    for i in range(args.runs):
        t = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(0)
        e = time.perf_counter() - t
        times.append(e)
        print(f"  run {i + 1}: {e:.2f} s")

    best = min(times)
    avg = sum(times) / len(times)
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device.type == "cuda" else float("nan")
    print("-" * 52)
    print(f"每样本: 最快 {best:.2f} s | 均值 {avg:.2f} s")
    if device.type == "cuda":
        print(f"显存峰值(模型+输入+激活): {peak:.0f} MiB（8 GB 卡可用上限约 8192 MiB）")
    print(f"500 样本推理预估: {best * 500 / 60:.1f} 分钟（按最快单次）")
    if device.type == "cuda" and peak > 6500:
        print("[警告] 显存峰值已接近 8 GB 上限，跑批时保持 batch=1，不要在循环里累积输出张量。")


if __name__ == "__main__":
    main()
