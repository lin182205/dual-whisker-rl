# Dual Whisker RL

Minimal Python project for dual-whisker active olfaction reinforcement learning.

For project goals, method route, current progress, and recommended next steps, see [PROJECT_RECORD.md](PROJECT_RECORD.md).

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
python scripts\random_rollout.py
python scripts\fixed_whisker_baseline_rollout.py
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4
python scripts\evaluate_joint_ppo.py --episodes 50
tensorboard --logdir results\tensorboard
```

移动机器人二维雷达与障碍闭环示例：

```powershell
python scripts\diagnose_mobile_lidar.py
python scripts\visualize_mobile_whisker_env.py --config configs\mobile_obstacles.yaml --no-model
python scripts\train_mobile_whisker_ppo.py --config configs\mobile_obstacles.yaml --timesteps 50000 --n-envs 4
```

障碍配置会把12维雷达距离加入策略观测，需从新模型开始训练；不传该配置时，现有无障碍观测维度保持不变。

VS Code workspace settings point Python to `.venv` and enable automatic virtual environment activation for new integrated terminals.

Generated figures are saved to `results/figures/`.
Evaluation scripts also save sensor-response curves, whisker-sector statistics, and an animated GIF of the first evaluated trajectory.
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
