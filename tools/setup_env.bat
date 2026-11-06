@echo off
REM ============================================================
REM CCUS-AI-Surrogate one-click environment setup (Windows)
REM What it does:
REM   1. create a virtualenv .venv
REM   2. upgrade pip
REM   3. install torch via Aliyun mirror (reachable everywhere; pulls the ~800MB CUDA build, which still runs fine on CPU-only laptops)
REM   4. install the rest (numpy/matplotlib/scikit-learn/pytest/streamlit) via Aliyun mirror
REM Usage: double-click this file, or run  tools\setup_env.bat  from the repo root
REM ============================================================

cd /d "%~dp0\.."

echo [1/4] Creating virtual environment .venv ...
python -m venv .venv
if errorlevel 1 (
  echo ERROR: python not found. Install Python 3.10+ first, then re-run.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

echo [2/4] Upgrading pip ...
python -m pip install --upgrade pip

echo [3/4] Installing torch (Aliyun mirror; CUDA build, runs on CPU too) ...
python -m pip install torch -i https://mirrors.aliyun.com/pypi/simple

echo [4/4] Installing remaining deps (Aliyun mirror) ...
python -m pip install numpy matplotlib scikit-learn pytest streamlit -i https://mirrors.aliyun.com/pypi/simple

echo.
echo Verifying imports ...
python -c "import torch,numpy,sklearn,matplotlib,streamlit,pytest; print('torch', torch.__version__, '| numpy', numpy.__version__, '| all imports OK')"

echo.
echo Environment ready. Next steps:
echo   python sim/make_dataset.py --n 800 --out data/synthetic.npz
echo   python -m src.train --model cnn --epochs 20
echo   python -m src.train_risk --epochs 30
echo   pytest tests/
echo   streamlit run src/app/streamlit_app.py
pause
