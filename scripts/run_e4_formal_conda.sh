#!/usr/bin/env bash
set -Eeuo pipefail

# E4 正式实验统一启动器。所有长任务默认携带 --resume：首次运行与断点恢复使用同一命令。

ENV_NAME="${DUAL_WHISKER_CONDA_ENV:-dual-whisker-rl}"
OUTPUT_DIR="${E4_OUTPUT_DIR:-results/e4}"
CONFIG_PATH="${E4_CONFIG_PATH:-configs/e4_formal.yaml}"
ACTION="run"
BACKGROUND=0
FOREGROUND_CHILD=0
LOG_FILE=""
RUNNER_ARGS=()

usage() {
    cat <<'EOF'
用法：bash scripts/run_e4_formal_conda.sh [阶段] [选项] [E4 参数]

阶段（默认 run）：
  run         校准、训练、评测、汇总；自动从断点恢复
  prepare     生成并校验正式场景
  calibrate   校准 B1 参数和 B2 固定角度；自动恢复
  train       只训练；自动从最近 checkpoint 恢复
  evaluate    只评测；自动跳过已完成场景
  summarize   重新汇总已有结果
  status      查看训练与评测状态
  stop        向当前正式实验发送 SIGINT，安全中断后可恢复

启动器选项：
  --env-name NAME      Conda 环境名，默认 dual-whisker-rl
  --output-dir PATH    E4 输出根目录，默认 results/e4
  --config PATH        formal 配置，默认 configs/e4_formal.yaml
  --background         使用 nohup 在后台运行；日志和 PID 自动保存
  --log-file PATH      指定日志文件
  -h, --help           显示帮助

其他参数原样传给 run_e4_experiments.py，例如：
  --methods main,b2_fixed --seeds 1,2 --n-envs 8

恢复说明：脚本始终传入 --resume。需要全新实验时，请换一个 --output-dir，
不要删除或覆盖原目录。中断后再次执行同一命令即可继续。
EOF
}

die() {
    echo "[正式实验启动失败] $*" >&2
    exit 1
}

if (($# > 0)) && [[ "$1" != -* ]]; then
    ACTION="$1"
    shift
fi
case "$ACTION" in
    run|prepare|calibrate|train|evaluate|summarize|status|stop) ;;
    *) die "未知阶段：$ACTION" ;;
esac

while (($# > 0)); do
    case "$1" in
        --env-name)
            (($# >= 2)) || die "--env-name 缺少参数"
            ENV_NAME="$2"
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir 缺少参数"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            (($# >= 2)) || die "--config 缺少参数"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=1
            shift
            ;;
        --foreground-child)
            FOREGROUND_CHILD=1
            shift
            ;;
        --log-file)
            (($# >= 2)) || die "--log-file 缺少参数"
            LOG_FILE="$2"
            shift 2
            ;;
        --profile|--resume)
            die "$1 由正式启动器固定管理，请不要重复传入"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            RUNNER_ARGS+=("$@")
            break
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "Conda 环境名格式无效：$ENV_NAME"
((BACKGROUND == 0 || FOREGROUND_CHILD == 0)) || die "后台子进程参数冲突"
[[ "$ACTION" != "status" && "$ACTION" != "stop" || $BACKGROUND -eq 0 ]] || die "$ACTION 不需要后台运行"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/run_e4_formal_conda.sh"
cd "$PROJECT_ROOT"

resolve_project_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$PROJECT_ROOT/$1"
    fi
}

OUTPUT_ABS="$(resolve_project_path "$OUTPUT_DIR")"
FORMAL_ROOT="$OUTPUT_ABS/formal"
PID_FILE="$FORMAL_ROOT/runner.pid"
mkdir -p "$FORMAL_ROOT" "$PROJECT_ROOT/results/logs/e4_formal"

if [[ "$ACTION" == "stop" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "当前没有正式实验 PID 记录：$PID_FILE"
        exit 0
    fi
    RUNNING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ ! "$RUNNING_PID" =~ ^[0-9]+$ ]] || ! kill -0 "$RUNNING_PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "PID 记录已失效并清理，没有进程需要停止。"
        exit 0
    fi
    if [[ -r "/proc/$RUNNING_PID/cmdline" ]]; then
        PROCESS_COMMAND="$(tr '\0' ' ' < "/proc/$RUNNING_PID/cmdline")"
        [[ "$PROCESS_COMMAND" == *"run_e4_experiments.py"* ]] || die "PID=$RUNNING_PID 不是 E4 运行器，拒绝发送信号"
    fi
    kill -INT "$RUNNING_PID"
    echo "已向 E4 进程发送 SIGINT：PID=$RUNNING_PID"
    echo "状态落盘后，可再次执行 bash scripts/run_e4_formal_conda.sh --background 恢复。"
    exit 0
fi

command -v conda >/dev/null 2>&1 || die "未找到 conda；请先运行 scripts/setup_server_conda.sh"
CONDA_BASE="$(conda info --base)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || die "未找到 Conda 初始化脚本：$CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME" || die "Conda 环境不存在：$ENV_NAME；请先运行部署脚本"
conda activate "$ENV_NAME"

python -c 'import stable_baselines3, tensorboard, torch' || die "环境依赖不完整；请重新运行 setup_server_conda.sh"
[[ -f "$CONFIG_PATH" ]] || die "正式配置不存在：$CONFIG_PATH"

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="results/logs/e4_formal/$(date '+%Y%m%d_%H%M%S')_${ACTION}.log"
fi
LOG_ABS="$(resolve_project_path "$LOG_FILE")"
mkdir -p "$(dirname -- "$LOG_ABS")"

if ((BACKGROUND == 1)); then
    CHILD_ARGS=("$ACTION" --env-name "$ENV_NAME" --output-dir "$OUTPUT_DIR" --config "$CONFIG_PATH" --log-file "$LOG_FILE" --foreground-child)
    CHILD_ARGS+=("${RUNNER_ARGS[@]}")
    nohup bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}" >/dev/null 2>&1 < /dev/null &
    LAUNCHER_PID=$!
    echo "正式实验已在后台启动：启动器 PID=$LAUNCHER_PID"
    echo "日志：$LOG_ABS"
    echo "状态：bash scripts/run_e4_formal_conda.sh status --output-dir '$OUTPUT_DIR'"
    exit 0
fi

exec > >(tee -a "$LOG_ABS") 2>&1
echo "============================================================"
echo "E4 正式实验阶段：$ACTION"
echo "Conda 环境：$ENV_NAME"
echo "输出目录：$FORMAL_ROOT"
echo "日志文件：$LOG_ABS"
echo "恢复策略：始终启用 --resume"
echo "启动时间：$(date --iso-8601=seconds)"
echo "============================================================"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "检测到运行中的实验进程：PID=$OLD_PID"
    elif [[ -n "$OLD_PID" ]]; then
        echo "检测到过期 PID 记录：$OLD_PID"
    fi
fi

if [[ "$ACTION" != "status" ]]; then
    command -v flock >/dev/null 2>&1 || die "系统缺少 flock，无法防止重复启动"
    exec 9>"$FORMAL_ROOT/.runner.lock"
    flock -n 9 || die "同一输出目录已有写任务运行；请先执行 status 检查"
fi

COMMAND=(python scripts/run_e4_experiments.py "$ACTION" --profile formal --config "$CONFIG_PATH" --output-dir "$OUTPUT_DIR" --resume)
COMMAND+=("${RUNNER_ARGS[@]}")
printf '执行命令：'
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "$ACTION" == "status" ]]; then
    exec "${COMMAND[@]}"
fi

"${COMMAND[@]}" &
CHILD_PID=$!
printf '%s\n' "$CHILD_PID" > "$PID_FILE"

forward_signal() {
    local signal="$1"
    if kill -0 "$CHILD_PID" 2>/dev/null; then
        kill "-$signal" "$CHILD_PID" 2>/dev/null || true
    fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

set +e
wait "$CHILD_PID"
EXIT_CODE=$?
set -e
if [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$CHILD_PID" ]]; then
    rm -f "$PID_FILE"
fi
if ((EXIT_CODE == 0)); then
    echo "阶段完成：$ACTION"
else
    echo "阶段中断或失败：$ACTION，退出码=$EXIT_CODE；再次执行同一命令将从完整 checkpoint/场景继续。" >&2
fi
exit "$EXIT_CODE"
