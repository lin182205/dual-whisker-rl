#!/usr/bin/env bash
set -Eeuo pipefail

# 双触须主动嗅觉项目：Linux + Conda 服务器一键部署。
# 默认只安装仿真、强化学习和 TensorBoard 依赖；真机与动画依赖按需启用。

ENV_NAME="${DUAL_WHISKER_CONDA_ENV:-dual-whisker-rl}"
PYTHON_VERSION="${DUAL_WHISKER_PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${DUAL_WHISKER_TORCH_INDEX_URL:-}"
CPU_ONLY=0
REQUIRE_CUDA=0
WITH_HARDWARE=0
WITH_FFMPEG=0
SKIP_CHECKS=0

usage() {
    cat <<'EOF'
用法：bash scripts/setup_server_conda.sh [选项]

选项：
  --env-name NAME          Conda 环境名，默认 dual-whisker-rl
  --python VERSION         Python 版本，默认 3.11
  --cpu-only              强制安装 CPU 版 PyTorch
  --require-cuda          检测到 torch.cuda 不可用时部署失败
  --torch-index-url URL   使用指定 PyTorch wheel 源
  --with-hardware         额外安装真机串口依赖
  --with-ffmpeg           通过 conda-forge 安装 FFmpeg
  --skip-checks           跳过部署后的 E4 与路径检查
  -h, --help              显示帮助

也可通过环境变量设置：
  DUAL_WHISKER_CONDA_ENV
  DUAL_WHISKER_PYTHON_VERSION
  DUAL_WHISKER_TORCH_INDEX_URL
EOF
}

die() {
    echo "[部署失败] $*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --env-name)
            (($# >= 2)) || die "--env-name 缺少参数"
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || die "--python 缺少参数"
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=1
            shift
            ;;
        --require-cuda)
            REQUIRE_CUDA=1
            shift
            ;;
        --torch-index-url)
            (($# >= 2)) || die "--torch-index-url 缺少参数"
            TORCH_INDEX_URL="$2"
            shift 2
            ;;
        --with-hardware)
            WITH_HARDWARE=1
            shift
            ;;
        --with-ffmpeg)
            WITH_FFMPEG=1
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数：$1"
            ;;
    esac
done

[[ "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "Conda 环境名只能包含字母、数字、点、下划线和连字符"
[[ "$PYTHON_VERSION" =~ ^[0-9]+([.][0-9]+){1,2}$ ]] || die "Python 版本格式无效：$PYTHON_VERSION"
((CPU_ONLY == 0)) || [[ -z "$TORCH_INDEX_URL" ]] || die "--cpu-only 与 --torch-index-url 不能同时使用"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

command -v conda >/dev/null 2>&1 || die "未找到 conda，请先确认服务器已安装 Conda 并加入 PATH"
[[ -f requirements-server.txt ]] || die "缺少 requirements-server.txt，请在项目根目录运行脚本"

export PYTHONUTF8=1
export MPLBACKEND=Agg
export PIP_DISABLE_PIP_VERSION_CHECK=1

conda_env_exists() {
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"
}

run_env() {
    conda run --no-capture-output -n "$ENV_NAME" "$@"
}

echo "[1/6] 项目目录：$PROJECT_ROOT"
echo "[2/6] Conda 环境：$ENV_NAME，Python：$PYTHON_VERSION"
if conda_env_exists; then
    echo "环境已存在，将复用并补齐依赖。"
else
    conda create --yes --name "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi
if ! run_env python -c "import sys; expected=tuple(map(int, '$PYTHON_VERSION'.split('.')[:2])); raise SystemExit(0 if sys.version_info[:2] == expected else 1)"; then
    die "环境 $ENV_NAME 的 Python 主次版本与 $PYTHON_VERSION 不一致；请改用新的 --env-name，或自行处理旧环境"
fi

echo "[3/6] 更新 Python 打包工具"
run_env python -m pip install --upgrade pip setuptools wheel

echo "[4/6] 安装 PyTorch 与项目依赖"
if ((CPU_ONLY == 1)); then
    run_env python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
elif [[ -n "$TORCH_INDEX_URL" ]]; then
    run_env python -m pip install --upgrade --index-url "$TORCH_INDEX_URL" "torch>=2.2"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "检测到 NVIDIA GPU，使用 requirements-server.txt 配置的 PyTorch 包。"
    echo "如需指定 CUDA wheel，请传入 --torch-index-url。"
else
    echo "未检测到 NVIDIA GPU，安装 CPU 版 PyTorch。"
    run_env python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
fi
run_env python -m pip install -r requirements-server.txt

if ((WITH_HARDWARE == 1)); then
    echo "安装真机串口依赖。"
    run_env python -m pip install -r requirements-hardware.txt
fi
if ((WITH_FFMPEG == 1)); then
    echo "通过 conda-forge 安装 FFmpeg。"
    conda install --yes --name "$ENV_NAME" --channel conda-forge ffmpeg
fi

echo "[5/6] 检查 Python、核心依赖与计算设备"
run_env env REQUIRE_CUDA="$REQUIRE_CUDA" python -c '
import os
import platform
import gymnasium
import matplotlib
import numpy
import stable_baselines3
import tensorboard
import torch
import yaml

print(f"Python: {platform.python_version()}")
print(f"NumPy: {numpy.__version__}")
print(f"Gymnasium: {gymnasium.__version__}")
print(f"PyYAML: {yaml.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Stable-Baselines3: {stable_baselines3.__version__}")
print(f"TensorBoard: {tensorboard.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"默认 GPU: {torch.cuda.get_device_name(0)}")
elif os.environ.get("REQUIRE_CUDA") == "1":
    raise SystemExit("要求 CUDA，但当前 PyTorch 无法使用 CUDA；请检查驱动和 PyTorch wheel 来源。")
'

echo "[6/6] 执行项目部署检查"
if ((SKIP_CHECKS == 0)); then
    run_env python -m unittest tests.test_e4_experiments
    run_env python scripts/check_portable_paths.py
    run_env python scripts/run_e4_experiments.py run \
        --profile formal \
        --dry-run \
        --output-dir results/setup_validation
else
    echo "已按要求跳过项目检查。"
fi

mkdir -p results/deployment
run_env python -m pip freeze > "results/deployment/${ENV_NAME}-pip-freeze.txt"

cat <<EOF

[部署完成]
激活环境：conda activate $ENV_NAME
Smoke 测试：python scripts/run_e4_experiments.py run --profile smoke --resume
正式实验：python scripts/run_e4_experiments.py run --profile formal --resume
TensorBoard：tensorboard --logdir results --host 127.0.0.1 --port 6006
依赖快照：results/deployment/${ENV_NAME}-pip-freeze.txt
EOF
