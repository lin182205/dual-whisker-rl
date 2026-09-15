# Dual Whisker RL

Minimal Python project for dual-whisker active olfaction reinforcement learning.

For project goals, method route, current progress, and recommended next steps, see [PROJECT_RECORD.md](PROJECT_RECORD.md).

项目架构图：

- [PCB 原理图接口规范（自研主控板权威输入）](docs/PCB_SCHEMATIC_INTERFACE_SPEC.md)
- [硬件模块图、接口与实验数据清单](docs/HARDWARE_MODULE_DIAGRAM.md)
- [Mobile 主线算法模块图与实验数据清单](docs/ALGORITHM_MODULE_DIAGRAM.md)
- [PPO 训练四模块详细流程图](docs/ALGORITHM_FOUR_DETAILED_DIAGRAMS.md)

The first milestone is a lightweight Gymnasium simulation with:

- a 2D odor plume;
- a mobile robot;
- dual virtual whisker sampling points;
- first-order slow-response gas sensors;
- a MultiDiscrete joint action space for movement and independent left/right whisker sectors;
- a fixed-whisker baseline environment where the policy controls movement only;
- a random rollout script that saves validation figures.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-rl.txt
python -m pip install -r requirements-hardware.txt
ffmpeg -version
python scripts\random_rollout.py
python scripts\fixed_whisker_baseline_rollout.py
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4
python scripts\evaluate_joint_ppo.py --episodes 50
tensorboard --logdir results\tensorboard
```

VS Code workspace settings point Python to `.venv` and enable automatic virtual environment activation for new integrated terminals.

FFmpeg is a system dependency for the default H.264 MP4 animation export. Install it
separately and make sure `ffmpeg` is available on `PATH`; it is not a Python package in
the requirements files. Explicit `.gif` output paths remain supported when a GIF is needed.

Generated figures are saved to `results/figures/`.
Evaluation scripts also save sensor-response curves, whisker-sector statistics, and an H.264 MP4 animation of the first evaluated trajectory.
Evaluation metrics include whisker information-gathering fields such as raw/sensor concentration, left-right contrast, plume contact ratio, odor loss duration, and left/right sector usage counts.

Training scripts write TensorBoard logs to `results/tensorboard/`. Common curves to watch are rollout episode reward, evaluation mean reward, and algorithm losses such as PPO value/policy loss or DQN TD loss.

Training seeds are random by default for better policy diversity. Pass `--seed 42` when you need a reproducible run. Each training run writes `run_metadata.json` under its log directory with the actual seed and config.

## Cloud training (Linux)

All CLI paths are repository-relative by default. The scripts resolve them against the
repository root, so they work both from the repository directory and when invoked by an
absolute script path from another working directory. `run_metadata.json` stores repository
paths such as `results/models/mobile_whisker_gru_ppo.zip`, without a Windows drive letter or
a cloud-machine home directory. Explicit absolute paths remain supported for mounted data disks.

```bash
git clone <repository-url> dual-whisker-rl
cd dual-whisker-rl

sudo apt-get update
sudo apt-get install -y ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-rl.txt

mkdir -p results/logs
nohup python scripts/train_mobile_whisker_ppo.py \
  --temporal-encoder gru \
  --timesteps 1000000 \
  --n-envs 8 \
  --seed 1 \
  > results/logs/mobile_cloud_stdout.log 2>&1 &
```

Inspect the process and logs:

```bash
tail -f results/logs/mobile_cloud_stdout.log
tensorboard --logdir results/tensorboard --host 127.0.0.1 --port 6006
```

To view TensorBoard locally, create an SSH tunnel with
`ssh -L 6006:127.0.0.1:6006 user@server`, then open `http://127.0.0.1:6006`.
The `results/` directory is intentionally ignored by Git; copy checkpoints and logs with
`rsync`, `scp`, or cloud object storage before releasing the server.

## E4 仿真对比实验

统一入口 `scripts/run_e4_experiments.py` 管理 M/B0–B6 的场景、训练、评估、恢复和汇总。默认 smoke 只用于连通性检查；正式结果使用固定验证/测试场景和 5 个训练种子。场地边界由配置文件的 `world_bounds: [-2.5, 2.5]` 控制，E4 smoke/formal 当前均使用该范围；旧 `world_half` 仍兼容。动态 puff 会按场地半宽和最小风速自动调整寿命，羽流网格、puff 裁剪和越界判断共享同一边界。训练环境后端由 `vec_env_backend` 控制：smoke 默认 `dummy` 便于调试，formal 默认 `subproc`，在 Windows 上以 `spawn` 启动 8 个独立环境进程；可用 `--vec-env-backend` 临时覆盖。

```powershell
# 生成 smoke 场景和任务清单
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py prepare --profile smoke
# 运行全部 smoke（可用 --methods main,b0_random 缩小范围）
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py run --profile smoke --resume
# 查看状态、重新评估或汇总
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py status --profile smoke
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py evaluate --profile smoke --resume
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py summarize --profile smoke
# 正式实验（先在验证集校准 B1 规则参数和 B2 十候选固定角度）
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py calibrate --profile formal
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py run --profile formal --resume
# 显式切换并行后端（auto 在 n-envs>1 时选择 subproc）
.\.venv\Scripts\python.exe scripts\run_e4_experiments.py train --profile formal --vec-env-backend auto --resume
```

每个方法×种子独立保存 `state.json`、模型、checkpoint 和逐场景 JSON；`summary.csv`、`summary.json`、`summary.md` 以及 `paired_differences.csv` 保留整体、分层和相对主方法差值。`--resume` 只复用版本、配置和场景摘要一致的完整产物。 汇总还显示每种子结果、期望回合数、覆盖率、缺失场景和配对缺失状态；重捕获无事件时耗时保持为空。
