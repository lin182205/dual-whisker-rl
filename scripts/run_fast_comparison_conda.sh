#!/usr/bin/env bash
set -Eeuo pipefail
# 仅调用无训练的首次到源对比入口；正式开机后自动选择并行配置。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DUAL_WHISKER_CONDA_ENV:-dual-whisker-rl}"
BACKGROUND=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--background" ]]; then BACKGROUND=1; else ARGS+=("$arg"); fi
done
cd "$ROOT"
if ((${#ARGS[@]}==0)) || [[ "${ARGS[0]}" == --* ]]; then ARGS=(run "${ARGS[@]}"); fi
COMMAND=(conda run --no-capture-output -n "$ENV_NAME" python -u scripts/run_fast_comparison.py "${ARGS[@]}")
if ((BACKGROUND)); then
    [[ "${ARGS[0]}" == run ]] || { echo '仅run支持后台运行'; exit 2; }
    mkdir -p results/logs/fast_comparison
    LOG="results/logs/fast_comparison/$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
    PID=$!
    echo "$PID" > results/logs/fast_comparison/latest.pid
    printf '%s\n' "$LOG" > results/logs/fast_comparison/latest.log
    printf '已提交后台任务，PID=%s；启动结果请查看日志：%s\n' "$PID" "$LOG"
else
    exec "${COMMAND[@]}"
fi
