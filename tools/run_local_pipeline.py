# -*- coding: utf-8 -*-
"""
CCUS 项目本地一键流水线（本机运行，非沙箱）。

按顺序完成：
  0. 准备目录 + 数据（data/synthetic.npz 缺失则自动生成）
  1. 确保 torch / streamlit 已装（缺则走阿里镜像装 CPU 版）
  2. P1 训练 CNN 代理模型
  3. P2 训练 FNO 代理模型
  4. P3 训练 RiskMLP 风险分级
  5. P1 可视化 真值 vs 预测
  6. P3 可视化 风险混淆矩阵
  7. P4 训练 UQ 代理集成（创新①）
  8. P4 训练 UQ 风险集成（创新①）

用法（在已激活的 .venv 中）：
  python tools/run_local_pipeline.py
或直接双击 tools/run_local_pipeline.bat（会自动激活 .venv）。
"""

import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PY = sys.executable  # 当前 venv 的 python，保证 pip/运行都用同一个环境


def log(msg):
    print(msg, flush=True)


def banner(title):
    log("\n" + "=" * 64)
    log(title)
    log("=" * 64)


def run(args, desc, critical=False):
    """运行一条子命令；返回 True/False。critical=True 时失败直接退出。"""
    log(f"\n>>> [{desc}]")
    log(f"$ {PY} {' '.join(args)}")
    t0 = time.time()
    try:
        r = subprocess.run([PY] + args)
    except Exception as e:  # noqa
        log(f"[FAIL] 启动失败：{e}")
        if critical:
            sys.exit(1)
        return False
    dt = time.time() - t0
    if r.returncode != 0:
        log(f"[FAIL] {desc} 退出码 {r.returncode}（耗时 {dt:.1f}s）")
        if critical:
            log("关键步骤失败，流水线终止。请检查上方报错后重试。")
            sys.exit(1)
        return False
    log(f"[OK] {desc} 完成（耗时 {dt:.1f}s）")
    return True


def ensure_dirs():
    for d in ("models", "experiments", "results"):
        os.makedirs(d, exist_ok=True)
    log("目录就绪：models/ experiments/ results/")


def ensure_data():
    data_path = os.path.join(ROOT, "data", "synthetic.npz")
    if os.path.exists(data_path):
        log(f"训练数据已存在：{data_path}（{os.path.getsize(data_path)/1e6:.1f} MB），跳过生成。")
        return
    log("data/synthetic.npz 缺失，自动生成（约 800 条样本，需几分钟）...")
    run(["-m", "sim.make_dataset", "--n", "800", "--out", "data/synthetic.npz"],
        "生成训练数据集", critical=True)


def ensure_torch():
    try:
        import torch  # noqa
        log(f"torch 已安装（{torch.__version__}），跳过安装。")
        return True
    except Exception:  # noqa
        pass
    log("torch 未安装，正在安装 torch(CPU 专用版, ~190MB) + streamlit...")
    # torch 走官方 CPU 专用源，避免 Windows 下从 PyPI 拉到 2-3GB 的 CUDA 版。
    ok = run(
        ["-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        "安装 torch(CPU)", critical=False,
    )
    if not ok:
        # 兜底：阿里镜像（会下到含 CUDA 的大包，但国内通常更快可达）
        log("官方 CPU 源失败，改用阿里镜像安装 torch（注意：会包含 CUDA，体积较大）...")
        ok = run(
            ["-m", "pip", "install", "torch",
             "-i", "https://mirrors.aliyun.com/pypi/simple/"],
            "安装 torch(阿里兜底)", critical=False,
        )
    # streamlit 走阿里镜像（纯 Python，无体积陷阱）
    run(
        ["-m", "pip", "install", "streamlit",
         "-i", "https://mirrors.aliyun.com/pypi/simple/"],
        "安装 streamlit", critical=False,
    )
    if not ok:
        log("torch 安装失败。请确认网络可访问 download.pytorch.org 或 mirrors.aliyun.com，")
        log("或手动执行：pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)
    return True


def main():
    banner("CCUS 项目本地一键流水线")
    log(f"项目根目录：{ROOT}")
    log(f"Python：{PY}")

    ensure_dirs()
    ensure_data()
    ensure_torch()

    steps = [
        (["-m", "src.train", "--model", "cnn", "--epochs", "20"],
         "P1 训练 CNN 代理模型"),
        (["-m", "src.train", "--model", "fno", "--epochs", "20"],
         "P2 训练 FNO 代理模型"),
        (["-m", "src.train_risk", "--epochs", "30"],
         "P3 训练 RiskMLP 风险分级"),
        (["-m", "src.plot_results", "fields",
          "--ckpt", "models/surrogate_cnn.pt", "--out", "results/pred_compare.png"],
         "P1 可视化 真值 vs 预测"),
        (["-m", "src.plot_results", "confusion",
          "--ckpt", "models/risk_mlp.pt", "--out", "results/confusion.png"],
         "P3 可视化 风险混淆矩阵"),
        (["-m", "src.train_uq", "--model", "cnn", "--seeds", "5", "--epochs", "20"],
         "P4 训练 UQ 代理集成（创新①）"),
        (["-m", "src.train_uq", "--risk", "--seeds", "5", "--epochs", "30"],
         "P4 训练 UQ 风险集成（创新①）"),
    ]

    t_all = time.time()
    ok = 0
    for args, desc in steps:
        if run(args, desc):
            ok += 1

    banner(f"流水线结束：成功 {ok}/{len(steps)} 步（总耗时 {time.time()-t_all:.1f}s）")

    log("\n预期产出清单：")
    expected = [
        "models/surrogate_cnn.pt",
        "models/surrogate_fno.pt",
        "models/risk_mlp.pt",
        "results/pred_compare.png",
        "results/confusion.png",
        "experiments/metrics_cnn.json",
        "experiments/metrics_fno.json",
        "experiments/metrics_risk.json",
        "experiments/metrics_uq_cnn.json",
        "experiments/metrics_uq_risk.json",
    ]
    for f in expected:
        p = os.path.join(ROOT, f)
        status = "OK " if os.path.exists(p) else "缺失"
        log(f"  [{status}] {f}")

    log("\n提示：")
    log(" - 缺失的文件对应失败的步骤，请查看上方对应步骤的报错。")
    log(" - 如需本地演示 App：streamlit run src/app/streamlit_app.py")
    log(" - 全部跑完后，把 experiments/*.json 发给你的助手评估 P1-P4 效果。")


if __name__ == "__main__":
    main()
