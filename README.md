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
python scripts\random_rollout.py
python scripts\fixed_whisker_baseline_rollout.py
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4
python scripts\evaluate_joint_ppo.py --episodes 50
tensorboard --logdir results\tensorboard
```

VS Code workspace settings point Python to `.venv` and enable automatic virtual environment activation for new integrated terminals.

Generated figures are saved to `results/figures/`.
Evaluation scripts also save sensor-response curves, whisker-sector statistics, and an animated GIF of the first evaluated trajectory.
Evaluation metrics include whisker information-gathering fields such as raw/sensor concentration, left-right contrast, plume contact ratio, odor loss duration, and left/right sector usage counts.

Training scripts write TensorBoard logs to `results/tensorboard/`. Common curves to watch are rollout episode reward, evaluation mean reward, and algorithm losses such as PPO value/policy loss or DQN TD loss.

Training seeds are random by default for better policy diversity. Pass `--seed 42` when you need a reproducible run. Each training run writes `run_metadata.json` under its log directory with the actual seed and config.
