# 训练 U-FNO 预测 CO2 气相饱和度 (Wen et al. 2022, Adv. Water Res.)
# 原 notebook: train_UFNO_gas_saturation.ipynb 的忠实转录，仅做三处改动：
#   1) device 改为可调（默认 cpu，适配无 GPU 笔记本）
#   2) 数据路径用 --data_dir 传入，不再写死 'your/data/path'
#   3) torch.load 加 weights_only=False，兼容 torch>=2.6 的默认安全策略
# 其余所有张量运算、mask、view 完全照搬官方 notebook，以保证与论文数值一致。
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from ufno import Net3d  # noqa: E402
from lploss import LpLoss  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train U-FNO for CO2 gas saturation (Wen 2022)")
    ap.add_argument("--data_dir", required=True,
                    help="含 sg_train_a.pt/sg_train_u.pt/sg_val_a.pt/sg_val_u.pt 的目录")
    ap.add_argument("--epochs", type=int, default=20, help="训练轮数（论文用 100，CPU 上先跑 2~20 验证管线）")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个训练样本（0=全部；本机内存/显存不够时用）")
    ap.add_argument("--save_dir", default="saved_models")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 数据加载 ----
    # ⚠️ 实测：sg_train_a.pt = (4500, 96, 200, 24, 12) ≈ 99.5GB，sg_train_u.pt ≈ 8.3GB。
    # mmap=True 才能在这种量级下不爆内存；本机（15.8GB 内存 / 47GB 空闲磁盘）实际放不下训练集，
    # 本脚本仅在有足够磁盘+内存+GPU 的机器上有意义，日常复现请走 eval 脚本 + 官方预训练权重。
    train_a = torch.load(f"{args.data_dir}/sg_train_a.pt", weights_only=False, mmap=True)
    train_u = torch.load(f"{args.data_dir}/sg_train_u.pt", weights_only=False, mmap=True)
    if args.limit > 0:
        train_a = train_a[:args.limit]
        train_u = train_u[:args.limit]
    print("train_a", tuple(train_a.shape), "train_u", tuple(train_u.shape), "device", device)

    mode1, mode2, mode3, width = 10, 10, 10, 36
    model = Net3d(mode1, mode2, mode3, width)
    model.to(device)

    # ---- 准备 x 方向导数所需的网格间距（照搬 notebook）----
    time_grid = np.cumsum(np.power(1.421245, range(24)))
    time_grid /= np.max(time_grid)
    grid_x = train_a[0, 0, :, 0, -3]
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2
    grid_dx = torch.tensor(grid_dx[None, None, :, None], dtype=torch.float32).to(device)

    epochs = args.epochs
    learning_rate = 0.001
    scheduler_step = 2
    scheduler_gamma = 0.9
    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)

    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        train_l2 = 0.0
        counter = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            dy = (y[:, :, 2:, :] - y[:, :, :-2, :]) / grid_dx

            optimizer.zero_grad()

            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)
            pred = model(x).view(-1, 96, 200, 24)
            dy_pred = (pred[:, :, 2:, :] - pred[:, :, :-2, :]) / grid_dx

            ori_loss = 0.0
            der_loss = 0.0
            mask_dy = mask[:, :, :198, :]
            for i in range(x.shape[0]):  # 末批可能不满 batch_size，按实际大小循环避免 IndexError
                ori_loss += myloss(pred[i, ...][mask[i, ...]].reshape(1, -1),
                                   y[i, ...][mask[i, ...]].reshape(1, -1))
                der_loss += myloss(dy_pred[i, ...][mask_dy[i, ...]].reshape(1, -1),
                                   dy[i, ...][mask_dy[i, ...]].view(1, -1))

            loss = ori_loss + 0.5 * der_loss

            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            if counter % 100 == 0:
                print(f"epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}")

        scheduler.step()
        print(f"epoch: {ep}, train loss: {train_l2/train_a.shape[0]:.4f}")

        if ep % 2 == 0:
            PATH = (f"{args.save_dir}/SG_UFNO_{ep}ep_{width}width_{mode1}m_"
                    f"{train_a.shape[0]}train_{learning_rate:.2e}lr.pt")
            torch.save(model, PATH)
            print(f"  saved -> {PATH}")


if __name__ == "__main__":
    main()
