# 服务器训练管理面板

空间探索筛选已加入面板：A保留旧奖励并屏蔽记忆，B加入访问记忆，C增加首次访问奖励，
D用近期空白区域重复扣分替代绕圈扣分。四组均为22.5厘米触须，从零各100万步。
观测、封存场景和自动复评规则见[空间探索筛选指南](docs/SPATIAL_EXPLORATION_PILOT.md)。

新增中文管理面板，可查看实时进度和曲线、创建训练或检查点续训、排队，以及
调整目标步数和保存后暂停/停止。现有两组训练默认只接入监控，手动确认后才接管。
Windows运行 `scripts/open_training_dashboard.ps1`，部署与控制语义见
[训练面板使用指南](docs/TRAINING_DASHBOARD.md)。

# 当前默认：办公场地训练与批量对比

新批量对比已使用当前 `small_office_v1`、连续底盘和离散触须，入口为
`scripts/run_red_experiments.py`（默认 `configs/office_experiments.yaml`）及
`scripts/run_e4_experiments.py`；风向与运动历史版结果分别写入
`results/office_history_experiments` 和 `results/office_history_e4`。方法、预算和旧实验用法见[办公批量实验指南](docs/OFFICE_BATCH_EXPERIMENTS.md)。

```powershell
python scripts/run_red_experiments.py plan --profile smoke
python scripts/run_red_experiments.py run --profile smoke --experiments e3 e5 --variants main,fixed,periodic --resume
```

## 既有正式实验：修正羽流与随机采样重评

既有模型的封存协议和服务器命令见[随机动作正式重评指南](docs/STOCHASTIC_REEVALUATION_ZH.md)。
先完成现有15M训练，再按修复后的物理参数和随机动作评估全部方法；旧模型、训练与结果保留。
E4每模型目标800万步，校准与正式训练默认先测实际吞吐，再选择最多两个模型并行。

# Dual Whisker RL

## Mobile 独立入口：连续底盘与双触须联合避障搜索

Mobile 独立训练、普通评估、固定场景评估和可视化默认使用
`configs/mobile_rooms_hybrid_history.yaml`，布局标识为 `small_office_v1`。
场地为 `[-5,5]×[-5,5]` 米，三间办公室、会议室、茶水间和卫生间沿公共通道组织，
下侧为接待前厅，主走廊净宽1.5米。墙厚0.1米、七个1米房门及一个1.6米开放口；
会议室分别通前厅与走廊，其他房间独立开门到公共区。左右隔墙错位、面积按用途区分。
只有墙体障碍。气源和机器人在包括走廊的自由区域随机初始化，机器人朝向均匀随机。
地图用于定位、碰撞和路径计算，不直接输入策略；房间编号R1–R6按实际区域判断，
走廊独立标记为 `corridor`、前厅为 `reception`。
研究参考、尺寸和门洞位置见[办公平面设计](docs/OFFICE_LAYOUT_DESIGN.md)。
运行 `python scripts/visualize_room_layout.py` 可从实际训练几何生成标注平面图。

动作按 `[linear_norm, angular_norm, left_sector, right_sector]` 打包，速度分量在
[-1,1]，映射到线速度[0,0.15]米/秒、角速度[-1.5,1.5]弧度/秒；触须为整数0–9。
项目内的混合 PPO 策略使用二维高斯和两个分类分布，联合对数概率与熵相加。
训练保留原始高斯样本；执行只限幅速度，触须直接使用分类结果。随机动作也遵守整数约束。

默认单帧 **57维**，顺序为原12维气味/触须特征、朝向cos/sin、实际线速度/v_max、
实际角速度/ω_max、blank_age、x/5与y/5、相对风向cos/sin、36维归一化雷达。
360条原始射线按36扇区取最小距离，量程3米；0号扇区以正前方居中（[-5°,5°)），
之后逆时针排列，角度首尾循环。风向表示气流运动方向，位姿和雷达默认无测量噪声。
20帧原始观测共 **1140维**。时序分支每帧 **19维**，包含13维气味、触须与空白时长，
2维风向及4维相对位姿。编码器用各帧的朝向把风向统一到当前车身坐标系，
并将历史位置相对当前位置的位移投影为车身前向、左向的米数，同时编码相对朝向的余弦和正弦。
导航分支只读取当前 **42维**，包括位置、朝向、实际速度和雷达。默认 GRU 与导航 MLP
各输出64维，拼成128维供策略与价值网络使用。`--temporal-encoder mlp/transformer`
可选择同样输入的历史编码器。地图位姿与风向仍来自仿真状态。
静态轨迹图和动画不显示雷达射线，雷达输入及逐步记录继续保留。
最终轨迹效果图不标注房间/区域名称或编号；用途、尺寸标注仅用于独立平面设计图。
羽流浓度图与色条统一使用浅蓝到深蓝的 `Blues` 色阶，高浓度端也为深蓝色。
气源采用红色星标，与蓝色浓度场区分；星标颜色不表示浓度。

机器人半径 0.15 米，碰墙取消平移并扣 0.5 分，保留转向且继续运行；触须采样遇墙截断。
底盘按差速圆弧积分，每子步位移≤1厘米、转角≤0.05弧度，并按弧弦偏离余量检测碰撞。
实际行程按圆弧长度累计，受阻步不计平移；日志分别保留请求与实际线/角速度。
动态 puff 中心遇墙反射，浓度计算屏蔽隔墙贡献，气体可通过门洞传播。该模型是轻量
几何约束近似，不求解压力或室内回流，也不对遮挡后的高斯质量作守恒重分配。
默认暖场 500 步、最多 1200 个 puff，寿命按场地尺度和最低风速计算。

每局最多 1200 步（240 秒），首次进入气源 0.1 米且无遮挡即成功。距离奖励为
`Phi(d_next)-Phi(d_previous)`，`Phi(d)=-8*clip(d/20,0,1)`；d 是按安全半径膨胀地图上的
最短可行路径距离，真实气源位置仅供奖励和评估。其他原有奖励系数保留。
表面间隙为 `max(0,min(lidar)-robot_radius)`，低于0.20米时成本为
`-0.05*clip(1-clearance/0.20,0,1)^2`；实际速度变化成本为
`-0.005*(abs(Δv)/v_max+abs(Δω)/(2*ω_max))`。成本可配置、单独记录；没有正的前进奖励。
参考 [DRL-robot-navigation 固定版本1adfe511](https://github.com/reiniscimurs/DRL-robot-navigation/tree/1adfe511d6f3ff5d7d3bdc357914501c29646c39)
的雷达分区、连续速度与近障成本思路，来源与适配说明见[第三方参考](docs/THIRD_PARTY_REFERENCES.md)。

```powershell
# 默认新实验名 mobile_hybrid_history_gru_ppo_<编号>，模型与 checkpoint 均保存环境契约
python scripts/train_mobile_whisker_ppo.py --timesteps 50000 --n-envs 4
# 以实际训练输出的模型路径为准
python scripts/evaluate_mobile_whisker_ppo.py --model-path results/models/mobile_hybrid_history_gru_ppo_1.zip --episodes 50 --no-visualization
python scripts/evaluate_mobile_fixed_scenarios.py --model-path results/models/mobile_hybrid_history_gru_ppo_1.zip --episodes 100 --workers 12
python scripts/visualize_mobile_whisker_env.py --no-model --steps 140
# 跨门洞下游诊断，输出间歇性、原始浓度和传感器响应图
python scripts/analyze_plume_intermittency.py --config configs/mobile_rooms_hybrid.yaml --source-position -2.5 -3.5 --wind-direction-deg 0 --downwind-distance 3 --crosswind-offsets 0 0.15 0.3 0.45 --threshold 0.2 --steps 3000 --label office
```

新模型/检查点会核对 `mobile_hybrid_nav_v1` 观测、`hybrid_motion_whisker_v1` 动作、
字段顺序、历史长度、物理速度范围、雷达分区、特征分组、编码器、地图摘要和有效配置；
续训严格匹配，评估仅允许改变回合步数上限。固定场景库使用独立房间版本和地图摘要，
封存后的场景可逐步复现原采样局，混合动作题库使用独立的mobile-hybrid版本和文件名。
两种评估的 `navigation` 字段统一给出碰撞、速度、最小表面间隙、近障步比例、控制变化、
可行路径距离、路径效率与同房间/跨房间分组。碰撞次数按受阻步计数，碰撞率为每局
碰撞步比例的平均值；路径效率仅对成功局计算，失败局保留为空。
同房/跨房分组按实际区域编号比较，走廊也视为独立区域，不再按坐标象限判断。

旧 15/16 维 Mobile 模型需显式传入其原训练配置，例如
`--config configs/mobile_whisker.yaml`；旧 15 维还需原配置的
`include_blank_age_observation: false`。旧模型不能用于新房间观测。
上一版56维四房间模型使用 `--config configs/mobile_rooms_four.yaml`，其地图摘要与
原四房间场景一致。六房间地图使用不同摘要，即使观测维度相同也不能混用模型和题库。
原 Mobile 5×5 配置及 E4/E5/E6 配置保留原定义和签名校验规则。
上一版56维六房间离散模型使用 `--config configs/mobile_rooms.yaml`；新旧动作和观测
不可混用，即使地图摘要相同也会拒绝加载。旧配置不会自动变成连续动作。
旧办公57维模型使用 `--config configs/mobile_rooms_hybrid.yaml`，旧六房间模型使用
`--config configs/mobile_rooms_hybrid_six.yaml`。新版编码契约 `wind_motion_history_v1`
区分原始维度相同的模型与场景清单。
它与当前办公图观测维度相同，但地图摘要不同，模型和固定题库不会混用。
连续底盘输出目前用于仿真，硬件触须 `STEP l r` 接口保持原样。

旧轨迹转换流程见[快速对比历史指南](docs/FAST_COMPARISON_ZH.md)。当前修正物理与随机采样协议使用上方正式重评指南，旧轨迹保留供溯源。

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

Mobile PPO 从头训练可用 `--experiment-name` 指定实验关键词（`--run-name` 同义）。
办公历史版默认关键词为 `mobile_hybrid_history_<encoder>_ppo`，旧连续房间为
`mobile_hybrid_<encoder>_ppo`，离散房间为 `mobile_rooms_<encoder>_ppo`，旧开放场为
`mobile_whisker_<encoder>_ppo`。每次从头训练自动分配下一个编号，例如
`odor_search_1`、`odor_search_2`；checkpoint、最终模型、日志与 TensorBoard 子目录
使用同一编号。指定 `--resume-from` 时，脚本从 checkpoint 恢复原实验名称、checkpoint
目录和 TensorBoard 记录，`--timesteps` 表示额外训练步数；此时忽略 `--experiment-name`。
旧 metadata 若未记录精确的 TensorBoard 子目录，默认沿用该实验编号最大的历史记录，
也可通过 `--tensorboard-run-dir` 显式指定。从头训练时，已存在的显式 `--model-path`
或 `--log-dir` 会被拒绝，防止覆盖。

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
（1×1）；Mobile 独立入口默认使用 configs/mobile_rooms.yaml（10×10 不规则六房间）。
configs/mobile_whisker.yaml 保留原5×5无障碍定义（边界 [-2.5, 2.5]），供旧模型和既有实验使用。
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


## 仿真对比实验入口

已完成旧六方法和75M主对比后，要先训练单帧MLP再补齐正式实验，请使用
[剩余正式实验执行指南](docs/REMAINING_FORMAL_ZH.md)。它在隔离的第2版封存环境中恢复旧实验，
不要求重复训练已完成的核心模型；默认新MLP1500万步，随后自动串联所有剩余阶段。

统一入口支持E3训练稳定性、E4主对比、E5消融和E6扰动测试。**未指定实验时默认全部运行**；
使用--experiments或--experiment选择一项或多项。详细命令、输出与统计说明见
[实验脚本中文使用指南](docs/SIMULATION_EXPERIMENT_GUIDE_ZH.md)。

~~~powershell
# 默认全部；plan只检查计划
.\.venv\Scripts\python.exe scripts\run_red_experiments.py plan --profile smoke
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --profile smoke --resume

# 手动选择；支持空格或逗号分隔
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e3 e6 --profile formal --resume

# 已有模型只评估，不训练
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 e6 --profile formal --model-path results/models/size_5-gru_3.zip --resume
~~~

每次执行在结果根目录的reports/<UTC时间戳>下保存独立报告，包含各实验的CSV、PNG预览、
PDF/SVG矢量图、图注与输入摘要；缺少真实证据时明确输出缺项状态表，不生成虚假曲线。
已有模型缺训练日志时不能重建E3，单个模型也不能替代E5的独立重训对照。

原生实验默认结果根为results/simulation_experiments/<profile>；已有模型默认根为
results/simulation_experiments/pretrained/<模型名>/<profile>。原E4保持独立协议，group/variants
只限制E3/E5/E6，原E4用--e4-methods选择方法。所有结果均为仿真，E0/E7真实实验延期。
