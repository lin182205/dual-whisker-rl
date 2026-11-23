#!/usr/bin/env bash
# 先激活训练环境。两组各训练1000万步，共享GPU，各占4个环境进程。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="$("${PYTHON_BIN:-python}" -c 'import sys; print(sys.executable)')"
command -v tmux >/dev/null
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "GPU尚未恢复，暂不启动两组训练"'
for session in office-pair-original office-pair-improved; do
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "训练会话已存在：$session" >&2
        exit 1
    fi
done
mkdir -p results
pair_root="$(mktemp -d "$project_root/results/office_pair_10m.XXXXXX")"
git rev-parse HEAD > "$pair_root/source_commit.txt"

for variant in original improved; do
    config=configs/mobile_rooms_hybrid_history.yaml
    if [[ "$variant" == improved ]]; then
        config=configs/mobile_rooms_hybrid_history_improved.yaml
    fi
    output="$pair_root/$variant"
    mkdir "$output"
    tmux new-session -d -s "office-pair-$variant" -c "$project_root" \
        bash -c '
            output=$1
            shift
            export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
            "$@" > "$output/train.log" 2>&1
            code=$?
            printf "%s\n" "$code" > "$output/exit_code"
            exit "$code"
        ' bash "$output" "$python_bin" -u scripts/train_mobile_whisker_ppo.py \
        --config "$config" --timesteps 10000000 --seed 11 \
        --temporal-encoder gru --history-length 20 \
        --n-envs 4 --vec-env-backend subproc --n-steps 1250 --batch-size 250 \
        --checkpoint-freq 100000 \
        --experiment-name "office_pair_${variant}_s11_10m" \
        --model-path "$output/final_model.zip" --log-dir "$output/logs" \
        --tensorboard-dir "$output/tensorboard"
done
printf '两组训练输出目录：%s\n' "$pair_root"
tmux list-sessions -F '#{session_name}'
