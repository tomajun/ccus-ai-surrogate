#!/usr/bin/env bash
# CCUS-AI-Surrogate 一键环境配置 (Linux / macOS)
# 作用：建 venv + 用 CPU 专用源装 torch（~190MB，避开 CUDA 版 ~800MB）+ 其余依赖
# 用法：chmod +x tools/setup_env.sh && ./tools/setup_env.sh

set -e
cd "$(dirname "$0")/.."

echo "[1/4] 创建虚拟环境 .venv ..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/4] 升级 pip ..."
python -m pip install --upgrade pip

echo "[3/4] 安装 torch (CPU 专用源，提速关键) ..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[4/4] 安装其余依赖 (numpy/matplotlib/scikit-learn/pytest/streamlit, 走阿里镜像提速) ..."
python -m pip install numpy matplotlib scikit-learn pytest streamlit -i https://mirrors.aliyun.com/pypi/simple

echo
echo "验证导入 ..."
python -c "import torch,numpy,sklearn,matplotlib,streamlit,pytest; print('torch', torch.__version__, '| numpy', numpy.__version__, '| 全部导入成功')"
echo
echo "环境就绪。后续运行示例："
echo "  python sim/make_dataset.py --n 800 --out data/synthetic.npz"
echo "  python -m src.train --model cnn --epochs 20"
echo "  pytest tests/"
echo "  streamlit run src/app/streamlit_app.py"
