# 评估 U-FNO：在测试集上计算"遮罩相对 L2 误差"，即论文 headline 指标。
# 路线：加载官方预训练权重做推理（不训练），因此只需要 sg_test_a.pt / sg_test_u.pt。
#
# 实测数据形状（2026-10-29 在本机 `数据集/` 上验证，与官方 notebook 输出一致）：
#   sg_test_a.pt = (500, 96, 200, 24, 12) float32   # (N, 竖向96, 水平200, 时间24, 通道12)
#   sg_test_u.pt = (500, 96, 200, 24)     float32   # CO2 气相饱和度，无通道维
#
# 官方权重是无扩展名文件（如 trained_model/sg_UFNO_4500_trained），整对象 pickle，
# 加载依赖 ufno.Net3d 可导入 —— 故本脚本先把自身目录塞进 sys.path，可从仓库根直接运行。
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台：别让一个生僻字符中断整轮评估

import torch

# 官方权重是整模型 pickle，类被记在 __main__；必须把 ufno 的全部模型类导入到本模块全局，
# 否则 torch.load 会报 Can't get attribute 'SimpleBlock3d' 等。
from ufno import Net3d, SimpleBlock3d, SpectralConv3d, U_net  # noqa: E402
from lploss import LpLoss  # noqa: E402
from ufno_reader import open_split  # noqa: E402


def _expose_to_main():
    """官方权重是在 notebook 里保存的整模型 pickle，类名解析成 __main__.Net3d。
    脚本自身就是 __main__ 时第 26 行的 import 已够用；从别的入口（-m / 别处 import）
    调用时不够，故显式把 ufno/lploss 的全部类注册进 __main__。"""
    import __main__
    import ufno as _ufno
    import lploss as _lploss
    for _mod in (_ufno, _lploss):
        for _name in dir(_mod):
            if isinstance(getattr(_mod, _name), type):
                setattr(__main__, _name, getattr(_mod, _name))


def main():
    ap = argparse.ArgumentParser(description="Eval U-FNO on CO2 gas saturation test set (inference only)")
    ap.add_argument("--model_path", required=True, help="官方预训练权重路径（可无扩展名）")
    ap.add_argument("--data_dir", required=True, help="含 sg_test_a.pt/sg_test_u.pt 的目录")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 个样本（0=全部 500）")
    ap.add_argument("--save_npz", default="", help="把 pred/y/mask 存成 npz，供三级预警校准阈值用")
    ap.add_argument("--metrics_out", default="", help="把指标写成 json")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[eval] device={device}")

    # 不用 torch.load(mmap=True)：本机实测该路径一旦读取/运算就 Segmentation fault。
    # 改用按偏移分页直读（见 ufno_reader），再逐样本落地成真实内存张量。
    test_a, test_u = open_split(args.data_dir, "test")
    print("test_a", tuple(test_a.shape), "test_u", tuple(test_u.shape), "dtype", test_a.dtype)
    assert test_a.shape[-1] == 12, f"输入通道应为 12，实得 {test_a.shape[-1]}"
    assert tuple(test_u.shape[1:]) == (96, 200, 24), f"输出形状应为 (N,96,200,24)，实得 {tuple(test_u.shape)}"

    # 官方权重是"整模型 pickle"，且是在 notebook 里保存的：类被绑定成 __main__.Net3d，
    # 直接 torch.load 会报 "Can't get attribute 'Net3d' on <module '__main__'>"。
    # 解法：先把本目录 ufno/lploss 里的所有类注册进 __main__，再反序列化。
    _expose_to_main()
    model = torch.load(args.model_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    myloss = LpLoss(size_average=False)
    n = test_a.shape[0] if args.limit <= 0 else min(args.limit, test_a.shape[0])
    rel_list = []
    preds = [] if args.save_npz else None
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            # np.array(...) 把 memmap 视图拷成真实内存；直接对 mmap 做 torch 运算会崩
            x = torch.from_numpy(np.array(test_a[i:i + 1])).to(device)
            y = torch.from_numpy(np.array(test_u[i:i + 1])).to(device)
            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)
            pred = model(x).view(-1, 96, 200, 24)
            e = myloss(pred[0, ...][mask[0, ...]].reshape(1, -1),
                       y[0, ...][mask[0, ...]].reshape(1, -1)).item()
            rel_list.append(e)
            if preds is not None:
                # 三个都要 .cpu()：GPU 张量不能直接 .numpy()
                preds.append((pred[0].cpu().numpy(), y[0].cpu().numpy(), mask[0].cpu().numpy()))
            if (i + 1) % 25 == 0 or i + 1 == n:
                rate = (time.time() - t0) / (i + 1)
                print(f"  [{i + 1}/{n}] mean_rel_L2={sum(rel_list)/(i+1):.4f} "
                      f"({rate:.1f}s/样本, 预计剩余 {rate*(n-i-1)/60:.1f} 分钟)")

    mean_rel = sum(rel_list) / len(rel_list)
    print(f"Test relative L2 (masked, mean over {n} samples): {mean_rel:.4f}")
    print("参照：官方 4500 训练样本的 U-FNO 权重在本测试集实测 mean rel L2 = 0.0647（500 样本，2026-10-30）。"
          "若你得到的值明显更大，优先怀疑：① 目标张量按 stride 还原了吗（u 是非连续转置视图）；"
          "② 权重与任务是否匹配（sg=气相饱和度，不是压力）。")

    if args.save_npz:
        p, yy, mm = zip(*preds)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez_compressed(args.save_npz,
                            pred=np.stack(p), y=np.stack(yy), mask=np.stack(mm),
                            rel_l2=np.array(rel_list, dtype=np.float32))
        print(f"[eval] 预测已存 -> {args.save_npz}")

    if args.metrics_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as fh:
            json.dump({"model_path": args.model_path, "n_samples": n,
                       "mean_rel_l2": float(mean_rel), "per_sample_rel_l2": [float(v) for v in rel_list],
                       "seconds_per_sample": float((time.time() - t0) / n)},
                      fh, ensure_ascii=False, indent=2)
        print(f"[eval] 指标已存 -> {args.metrics_out}")


if __name__ == "__main__":
    main()
