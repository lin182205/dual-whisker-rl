# Dual Whisker RL

Minimal Python project for dual-whisker active olfaction reinforcement learning.

The first milestone is a lightweight Gymnasium simulation with:

- a 2D odor plume;
- a mobile robot;
- dual virtual whisker sampling points;
- first-order slow-response gas sensors;
- a discrete joint action space for movement and sensing;
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
python scripts\train_joint_dqn.py --timesteps 50000
python scripts\evaluate_joint_dqn.py --episodes 50
```

VS Code workspace settings point Python to `.venv` and enable automatic virtual environment activation for new integrated terminals.

Generated figures are saved to `results/figures/`.
Evaluation scripts also save an animated GIF of the first evaluated trajectory.
