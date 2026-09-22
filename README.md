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
python scripts\train_mobile_whisker_ppo.py
python scripts\evaluate_mobile_fixed_scenarios.py --prepare-only
python scripts\evaluate_mobile_fixed_scenarios.py --workers 12
tensorboard --logdir results\tensorboard
```

本地 IDE 配置不进入版本库。使用 VS Code 时，请在本机选择 `.venv` 中的 Python 解释器；服务器按下方 Linux 命令单独创建环境。

FFmpeg is a system dependency for the default H.264 MP4 animation export. Install it
separately and make sure `ffmpeg` is available on `PATH`; it is not a Python package in
the requirements files. Explicit `.gif` output paths remain supported when a GIF is needed.

Generated figures are saved to `results/figures/`.
Evaluation scripts also save sensor-response curves, whisker-sector statistics, and an H.264 MP4 animation of the first evaluated trajectory.
Evaluation metrics include whisker information-gathering fields such as raw/sensor concentration, left-right contrast, plume contact ratio, odor loss duration, and left/right sector usage counts.

Training scripts write TensorBoard logs to `results/tensorboard/`. Mobile PPO does not run
synchronous evaluation during training; watch its rolling 100/500-episode success rates,
final distance, out-of-bounds rate, episode reward, and PPO losses. Fixed-scenario evaluation
is a separate multi-process command after checkpoints are available.

Training seeds are random by default for better policy diversity. Pass `--seed 42` when you need a reproducible run. Each training run writes `run_metadata.json` under its log directory with the actual seed and config.

Mobile PPO 独立训练可用 `--experiment-name` 指定实验关键词（旧参数 `--run-name` 同义）。
默认关键词为 `mobile_whisker_<encoder>_ppo`。每次启动自动分配下一个编号，例如
`odor_search_1`、`odor_search_2`；checkpoint、最终模型、日志与 TensorBoard 子目录
都使用该编号。从 `--resume-from` 续训也会分配新编号，并保留来源 checkpoint。
显式传入的 `--model-path` 或 `--log-dir` 若已存在，脚本会拒绝覆盖。

```powershell
python scripts\train_mobile_whisker_ppo.py --experiment-name odor_search
```

独立 PPO 训练入口默认使用 `--vec-env-backend auto`：当 `--n-envs` 大于 1 时，
每个环境场景运行在独立 Python 进程中；单环境自动回退到同进程。调试时可显式传
`--vec-env-backend dummy`，需要强制子进程时传 `--vec-env-backend subproc`。
移动 GRU/PPO 独立训练默认使用 12 个并行环境执行 8,000,000 步；每个完整 PPO
轮次开始和结束时会打印纯训练与含 checkpoint 的实际预计剩余时间。训练期间不再暂停执行
evaluation；`evaluate_mobile_fixed_scenarios.py` 首次运行会封存默认 100 个场景，后续可用
默认 12 个进程对不同 checkpoint 进行完全相同的确定性复评。

## Cloud training (Linux)

服务器最小上传文件、依赖分组和可排除目录见
[服务器训练文件与依赖清单](docs/SERVER_TRAINING_MANIFEST.md)。仿真服务器可用一个入口安装依赖：

```bash
bash scripts/setup_server_conda.sh
conda activate dual-whisker-rl
```

脚本会创建或复用 Conda 环境，安装 PyTorch、SB3、TensorBoard 和仿真依赖，随后检查核心包、CUDA、E4 单元测试、路径可移植性及 formal dry-run。无 NVIDIA GPU 时默认安装 CPU 版 PyTorch；GPU 服务器可加 `--require-cuda`，自定义 PyTorch wheel 源可传 `--torch-index-url URL`。真机或动画环境分别加 `--with-hardware`、`--with-ffmpeg`。

All CLI paths are repository-relative by default. The scripts resolve them against the
repository root, so they work both from the repository directory and when invoked by an
absolute script path from another working directory. `run_metadata.json` stores repository
paths such as `results/models/mobile_whisker_gru_ppo.zip`, without a Windows drive letter or
a cloud-machine home directory. Explicit absolute paths remain supported for mounted data disks.

```bash
git clone https://github.com/lin182205/dual-whisker-rl.git
cd dual-whisker-rl
bash scripts/setup_server_conda.sh --require-cuda
conda activate dual-whisker-rl

mkdir -p results/logs
bash scripts/run_e4_formal_conda.sh --background
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

上传服务器前可运行路径审计；它会检查代码和配置中的 Windows 盘符、用户目录、
Linux/macOS 家目录、挂载目录和 UNC 路径字面量：

```powershell
.\.venv\Scripts\python.exe scripts\check_portable_paths.py
```

图表脚本会自动发现系统中文字体。服务器没有中文字体时可安装 Noto CJK，或通过
`DUAL_WHISKER_FONT_REGULAR`、`DUAL_WHISKER_FONT_BOLD` 指定字体文件；训练过程不依赖这些字体。
建议通过 Git 克隆或归档源码上传。若直接复制工作区，应排除 `.venv`、`results`、
STM32 `build`、`.git` 和本地 IDE 设置目录；这些环境或生成产物可能包含本机路径，但不属于训练源码。

## 独立仿真入口的场景配置

独立训练与评估通过同一份 YAML 定义环境场景：静态高斯场的 Joint PPO / Fixed-whisker DQN 使用
configs/default.yaml（10×10）；固定机器人触须实验使用 configs/whisker_only.yaml
（1×1）；移动动态 puff 实验使用 configs/mobile_whisker.yaml（5×5，边界 [-2.5, 2.5]）。
对应的训练、评估、固定场景复评和 Mobile 可视化入口默认读取各自配置；--config
可显式选择另一份配置，命令行场景参数仅在显式传入时覆盖 YAML。触须采样对比的
--steps 默认取配置中的 max_steps。

Mobile 固定场景库默认按有效配置摘要、基础种子和局数命名；复用时也会核对这些参数。
评估 JSON 保存有效环境配置，便于核对训练 run_metadata.json。已有模型仍保留训练时的场景：
旧 Mobile checkpoint 若记录空配置，代表当时环境默认的 2×2；评估旧模型时应显式核对
metadata，并根据实验目的传入对应的 --config。E4 使用自己的 smoke/formal 配置和固定场景库。

## E4 仿真对比实验

统一入口 `scripts/run_e4_experiments.py` 管理 M/B0–B6 的场景、训练、评估、恢复和汇总。默认 smoke 只用于连通性检查；正式结果使用固定验证/测试场景和 5 个训练种子。场地边界由配置文件的 `world_bounds: [-2.5, 2.5]` 控制，E4 smoke/formal 当前均使用该范围；旧 `world_half` 仍兼容。动态 puff 会按场地半宽和最小风速自动调整寿命，羽流网格、puff 裁剪和越界判断共享同一边界。训练环境后端由 `vec_env_backend` 控制：smoke 默认 `dummy` 便于调试，formal 默认 `subproc`，在 Windows 上以 `spawn` 启动 8 个独立环境进程；可用 `--vec-env-backend` 临时覆盖。训练期间每完成一个“环境 rollout + 5 个 PPO epoch 更新”周期，终端会根据最近 5 个周期的墙钟时间打印预计剩余分钟数。

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

服务器正式实验推荐使用 Conda 启动器。它固定 formal 配置并始终添加 `--resume`，首次运行和断点恢复使用同一条命令：

```bash
# 前台运行；SSH 会话应配合 tmux/screen
bash scripts/run_e4_formal_conda.sh
# 后台运行，断开 SSH 后继续
bash scripts/run_e4_formal_conda.sh --background
# 查看任务与评测进度
bash scripts/run_e4_formal_conda.sh status
# 安全中断当前任务
bash scripts/run_e4_formal_conda.sh stop
# 中断后恢复：再次执行原命令
bash scripts/run_e4_formal_conda.sh --background
```

脚本用文件锁阻止同一输出目录被重复启动，自动保存日志和运行 PID。只续训可执行 `bash scripts/run_e4_formal_conda.sh train --background`，只补评测可执行 `... evaluate --background`。需要完全独立的新实验时传入新的 `--output-dir`；不要删除旧 checkpoint。正式校准产物也带场景、配置和代码签名，恢复时匹配则跳过 B1/B2 已完成搜索，不兼容时明确拒绝复用。首次校准会持续打印 B1/B2 候选进度、验证场景进度和预计剩余分钟。
