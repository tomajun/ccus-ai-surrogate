#!/usr/bin/env bash
# 等 tools/download_torch_cuda.sh 下完 wheel -> 离线装进 .venv -> 验证 CUDA -> 跑 GPU 基准
# 全程输出到 tools/_finish_cuda.log，后台执行，避免前台超时把进程带走。
set -u
ROOT='C:/workplace/ccus-ai-surrogate'
WHL="C:/workplace/.wheels/torch-2.15.0+cu132-cp313-cp313-win_amd64.whl"
PY="C:/workplace/ccus-ai-surrogate/.venv/Scripts/python.exe"
EXPECT=2108256835
LOG="$ROOT/tools/_finish_cuda.log"
exec >"$LOG" 2>&1

# 已经装好就直接收工（定时任务反复调用也安全）
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[finish] $(date) torch CUDA 已可用，无需处理"
  "$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
  exit 0
fi

# 单实例锁：避免两个 watcher 同时装
LOCK='/d/Temp/qoder-finish-cuda.lock'
if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) )) -lt 1800 ]; then
  echo "[finish] 已有 watcher 在跑（锁 $LOCK 未过期），本次退出"
  exit 0
fi
echo $$ > "$LOCK"

echo "[finish] $(date) 开始等待下载完成"
for i in $(seq 1 720); do
  HAVE=$(stat -c %s "$WHL" 2>/dev/null || echo 0)
  if [ "$HAVE" -ge "$EXPECT" ]; then
    echo "[finish] 下载字节数达标 $HAVE/$EXPECT"
    break
  fi
  echo "[finish] 轮询 $i: $HAVE/$EXPECT"
  touch "$LOCK"
  sleep 15
done

HAVE=$(stat -c %s "$WHL" 2>/dev/null || echo 0)
if [ "$HAVE" -lt "$EXPECT" ]; then
  echo "[finish] 超时未完成：$HAVE/$EXPECT，终止（重跑 tools/run_finish_cuda.sh 即可续传）"
  exit 1
fi

# 校验 sha256：曾出现两个 curl 同时写同一个文件，必须证明文件完好再安装
echo "[finish] 校验 wheel sha256"
EXPECTED_SHA=$(curl -s --max-time 60 https://download.pytorch.org/whl/cu132/torch/ \
  | tr '>' '\n' | grep -o "torch-2\.15\.0%2Bcu132-cp313-cp313-win_amd64\.whl#sha256=[0-9a-f]*" \
  | head -1 | sed 's/.*sha256=//')
ACTUAL_SHA=$(sha256sum "$WHL" | awk '{print $1}')
echo "[finish] 期望 $EXPECTED_SHA"
echo "[finish] 实际 $ACTUAL_SHA"
if [ -n "$EXPECTED_SHA" ] && [ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]; then
  echo "[finish] 校验失败：文件损坏，删除后重新下载"
  rm -f "$WHL"
  bash "$ROOT/tools/download_torch_cuda.sh"
  echo "[finish] 重下后退出，请再次运行 run_finish_cuda.sh"
  exit 1
fi

echo "[finish] 离线安装 wheel（先直接覆盖装，失败才 uninstall 后重装，避免中途失去 torch）"
install_wheel() {
  "$PY" -m pip install --no-index --force-reinstall --no-deps "$WHL"
  echo "[finish] PIP_EXIT=$?"
}
verify() {
  "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0), "| cap:", torch.cuda.get_device_capability(0))
    print("mem:", [m / 2**20 for m in torch.cuda.mem_get_info()], "MiB (free/total)")
    raise SystemExit(0)
raise SystemExit(1)
EOF
  echo "[finish] VERIFY_EXIT=$?"
}
install_wheel && verify
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[finish] 覆盖装后 CUDA 不可用，改为 uninstall 后干净重装"
  "$PY" -m pip uninstall -y torch
  install_wheel && verify
fi

echo "[finish] GPU 推理基准"
"$PY" "$ROOT/tools/bench_gpu_inference.py" --data_dir "$ROOT/数据集" --device cuda
echo "[finish] BENCH_EXIT=$?"
echo "[finish] $(date) 全部结束"
