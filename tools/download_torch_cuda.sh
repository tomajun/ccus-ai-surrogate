#!/bin/bash
# 断点续传下载 torch CUDA wheel，直到字节数与 Content-Length 一致；每轮打印进度。
set -u
URL='https://download-r2.pytorch.org/whl/cu132/torch-2.15.0%2Bcu132-cp313-cp313-win_amd64.whl'
OUT_DIR='/c/workplace/.wheels'
OUT="$OUT_DIR/torch-2.15.0+cu132-cp313-cp313-win_amd64.whl"
mkdir -p "$OUT_DIR"

EXPECT=$(curl -sIL --max-time 60 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
echo "[dl] 期望字节数 = ${EXPECT:-未知}"

for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  HAVE=0
  [ -f "$OUT" ] && HAVE=$(stat -c%s "$OUT")
  echo "[dl] 第 $attempt 轮，已有 $HAVE 字节，继续下载..."
  # -C - 从断点续传；速度连续 60s 低于 20KB/s 则放弃本轮重来
  curl -L -C - -o "$OUT" --retry 5 --retry-delay 3 \
       --speed-time 60 --speed-limit 20000 --max-time 1800 "$URL" 2>>"$OUT_DIR/dl.log"
  HAVE=0
  [ -f "$OUT" ] && HAVE=$(stat -c%s "$OUT")
  echo "[dl] 第 $attempt 轮结束，现有 $HAVE 字节"
  if [ -n "${EXPECT:-}" ] && [ "$HAVE" = "$EXPECT" ]; then
    echo "[dl] DONE 大小吻合: $HAVE"
    exit 0
  fi
  if [ -z "${EXPECT:-}" ] && [ "$HAVE" -gt 2000000000 ]; then
    echo "[dl] DONE (无 Content-Length，按体积判断): $HAVE"
    exit 0
  fi
done
echo "[dl] FAILED 12 轮仍未下完，现有 ${HAVE:-0} / ${EXPECT:-?} 字节"
exit 1
