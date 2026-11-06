#!/usr/bin/env bash
# 脱离式启动"等下载完 -> 校验 -> 装 CUDA torch -> 验证 -> GPU 基准"。
# 不挂靠在当前会话上：会话/应用重启也不会把下载后的收尾流程带走。
# 反复执行安全（脚本内部有单实例锁 + 已装好直接退出）。
set -u
ROOT='C:/workplace/ccus-ai-surrogate'
if command -v wmic >/dev/null 2>&1; then
  wmic process call create "cmd.exe /c call \"C:/Program Files/Git/bin/bash.exe\" -lc \"$ROOT/tools/finish_cuda_install.sh\"" 2>&1 | tr -d '\r' | grep -Eo "ProcessId\s+[0-9]+" || echo "[run] 已提交（未回显 PID）"
else
  powershell -NoProfile -Command "Start-Process -FilePath 'C:/Program Files/Git/bin/bash.exe' -ArgumentList '-lc', '\"$ROOT/tools/finish_cuda_install.sh\"' -WindowStyle Hidden" >/dev/null
  echo "[run] 已用 Start-Process 提交"
fi
echo "[run] 进度看 $ROOT/tools/_finish_cuda.log"
