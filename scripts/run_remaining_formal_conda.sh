#!/usr/bin/env bash
set -Eeuo pipefail
# 一个入口串联全部阶段；后台作业与SSH会话脱离，Python负责锁、监控和恢复。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DUAL_WHISKER_CONDA_ENV:-dual-whisker-rl}"
ACTION=run
BACKGROUND=0
ARGS=()
if (($#)) && [[ "$1" != --* ]]; then ACTION="$1"; shift; fi
for arg in "$@"; do
    if [[ "$arg" == --background ]]; then BACKGROUND=1; else ARGS+=("$arg"); fi
done
case "$ACTION" in plan|run|status) ;; *) echo '阶段仅支持plan/run/status'; exit 2 ;; esac
cd "$ROOT"
COMMAND=(conda run --no-capture-output -n "$ENV_NAME" python -u scripts/run_remaining_formal.py "$ACTION")
if [[ "$ACTION" == run ]]; then COMMAND+=(--resume); fi
COMMAND+=("${ARGS[@]}")
if ((BACKGROUND)); then
    [[ "$ACTION" == run ]] || { echo '仅run支持后台运行'; exit 2; }
    mkdir -p results/logs/remaining_formal
    LOG="results/logs/remaining_formal/$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
    printf '%s\n' "$!" > results/logs/remaining_formal/latest.pid
    printf '%s\n' "$LOG" > results/logs/remaining_formal/latest.log
    printf '后台任务已提交，PID=%s；需查看日志确认启动：%s\n' "$!" "$LOG"
else
    exec "${COMMAND[@]}"
fi
