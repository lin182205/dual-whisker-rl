# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目是什么

双触须主动嗅觉强化学习。研究问题**不是**沿浓度梯度爬到气源，而是验证：在间歇羽流、MQ-3 慢响应的条件下，**主动控制两根触须上的气体传感器采样位置**，能否比固定传感器获得更多有效气味信息（命中率、响应趋势、左右对比、重新捕获能力）。项目横跨仿真（Gymnasium + Stable-Baselines3）和真实硬件（PC ↔ STM32F103 ↔ 2 舵机 + 2 个 MQ-3，串口通信）。

Python 代码中的注释和 docstring **基本是中文**；编辑既有文件时请保持一致。

## 权威文档（做非平凡改动前先读）

- `PROJECT_RECORD.md` —— 活的设计/路线 + 进度日志，按章节**追加式**记录（目前到 §17）。记录决策的*原因*、当前理解、各阶段结论。**先读最新的几节。**
- `PROJECT_FEASIBILITY_AND_ROADMAP.md` —— 工程路线图：6 阶段计划、风险分析、实验决策树、奖励设计指南（§10.5）、当前*不该做*的事。
- `PROJECT_SHORT_TERM_PLAN.md`、`README.md` —— 快速上手 / 近期任务。

改动环境动力学、奖励、观测或羽流模型后，请更新 `PROJECT_RECORD.md` 对应章节 —— 团队依赖这份文档恢复上下文。

## 常用命令

环境配置（以下为 Windows PowerShell；项目运行在 `.venv` 上）：
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt          # 仿真核心：gymnasium, numpy, matplotlib, pyyaml
python -m pip install -r requirements-rl.txt       # 训练：torch, stable-baselines3, tensorboard
python -m pip install -r requirements-hardware.txt # 硬件：pyserial
```

仓库**没有测试套件、linter 或构建配置**（无 pytest/ruff/pyproject）。验证一处改动 = 跑相应的 rollout / 诊断 / smoke 训练脚本，看它输出的图和指标。

常用运行（路径用 `scripts\...`；正斜杠也可以）：
```powershell
# 环境快速 smoke（图保存到 results/figures/）
python scripts\random_rollout.py
python scripts\visualize_whisker_only_env.py --steps 140 --interpolation-frames 2

# 训练 / 评估（每次训练在日志目录写 run_metadata.json）
python scripts\train_fixed_whisker_dqn.py --timesteps 50000        # DQN 基线（Discrete 移动）
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4     # PPO 联合控制（MultiDiscrete）
python scripts\train_whisker_only_ppo.py --timesteps 50000 --n-envs 4 --history-length 6 [--domain-randomization]
python scripts\evaluate_joint_ppo.py --episodes 50
tensorboard --logdir results\tensorboard

# 羽流真实度诊断 —— 任何羽流改动前后都跑一次，对比
python scripts\analyze_plume_intermittency.py --label before --steps 3000 --wind-mode fixed

# 硬件（需先烧录 STM32 并接好串口）
python scripts\hardware_smoke_test.py --port COM3
python scripts\analyze_hardware_baseline.py            # 离线验证预处理
python scripts\compare_hardware_sampling_modes.py      # fixed / periodic / random 采样对比
```

种子默认随机（提高策略多样性）；需要复现时传 `--seed N`。`results/` 已 gitignore —— 不要提交模型、日志、图像。

## 架构

### 三个仿真环境（`dual_whisker_rl/envs/`），都是 Gymnasium
它们构成实验层级（主方法 vs 基线）：

- **`PlumeEnv`**（`plume_env.py`）—— 主方法，联合控制。动作 `MultiDiscrete([6, 10, 10])` = 机器人移动 + 左扇区 + 右扇区。用 **PPO** 训练。使用静态 `GaussianPlume`。
- **`FixedWhiskerPlumeEnv`**（`fixed_whisker_env.py`）—— `PlumeEnv` 的子类；动作降为 `Discrete(6)`（只控移动，触须自动扫描）。公平性基线。用 **DQN** 训练。
- **`WhiskerOnlyPuffEnv`**（`whisker_only_env.py`）—— 机器人固定，动作 `MultiDiscrete([10, 10])`（仅左右扇区）。**面向 sim-to-real**、当前开发最活跃的环境。使用动态 `DynamicPuffPlume`（定义在同一文件内）和 `AsymmetricGasSensor`；支持域随机化。

算法选择由动作空间决定：`MultiDiscrete` 用 **PPO**，单一 `Discrete` 才用 **DQN**。

### 共享组件模型（`dual_whisker_rl/envs/`）
环境是组合这些组件，而非各自重写物理：
- `plume_model.py::GaussianPlume` —— 静态下风向高斯场（`PlumeEnv` 用）。
- `whisker_only_env.py::DynamicPuffPlume` —— Farrell 家族的**丝状 filament/puff 羽流**：从上风侧气源离散释放 puff，随风平流，叠加 per-puff OU 湍流 + 风向 OU **蜿蜒（meander）**，产生真实的 whiff/blank **间歇性**。带 `domain_randomization` 每个 episode 扰动物理参数。这是决定真实度的关键模型。
- `robot_model.py` —— `DifferentialDriveRobot`（移动动作集）+ `RobotState`。
- `whisker_model.py::DualWhiskerSampler` —— 把扇区编号 → 触须角度 → 采样点，带**舵机角速度限制**，使触须分多步转向目标而非瞬移（贴近真实舵机）。
- `sensor_model.py` —— `FirstOrderGasSensor`（对称 EMA 滞后）和 `AsymmetricGasSensor`（响应快/恢复慢 + 可选基线漂移 + 噪声，模拟真实 MQ-3）。`WhiskerOnlyPuffEnv` 默认用非对称那个。

### 仿真↔硬件契约（最关键的跨层思想）
触须动作在各处刻意保持同一抽象：`WhiskerOnlyPuffEnv` 的 `[left_sector, right_sector]` 与串口命令 `STEP left_sector right_sector` 一一对应。仿真训练出的策略不需重新设计动作接口就能驱动真机——只有*观测*需要用硬件可获得的特征重建。

- `dual_whisker_rl/hardware/serial_client.py::DualWhiskerSerialClient` —— 发送 `STEP l r\n`，读回一行 JSON（`{"left_sector","right_sector","left_adc","right_adc"}`）。
- `dual_whisker_rl/hardware/sensor_preprocess.py` —— `OnlineGasPreprocessor` / `DualGasPreprocessor`：把原始 MQ-3 ADC 转成扣基线 / EMA 平滑 / 趋势 / 归一化特征（含左右差分）。这是**观测适配层**，让仿真策略能扛住真实传感器漂移；仿真观测也应逐步靠拢同样的特征形式。
- `hardware/firmware/stm32_dual_whisker/` —— STM32CubeIDE / HAL 工程（UART 命令解析、TIM PWM 舵机、ADC）。在 CubeIDE 里编译烧录，**不**经本仓库的 Python 工具链。

### 评估、可视化、训练胶水
- `evaluation.py` —— 共享评估循环 + 信息获取指标（成功率、回报、气味命中、重新捕获、扇区使用）。两个评估脚本的输出字段应保持一致。
- `visualization.py`、`plotting.py` —— 羽流场、轨迹、传感器响应曲线、触须角度曲线、轨迹 GIF（输出到 `results/figures/`）。
- `ObservationHistoryWrapper`（定义在 `scripts/train_whisker_only_ppo.py` 内）堆叠最近 N 帧观测，让 PPO 看到慢响应传感器的短期趋势——因为单帧 MQ-3 不足以体现其动态。

## 工作流约定（来自 PROJECT_RECORD §8）

- 改动环境动力学 / 奖励 / 观测后：对受影响的环境重跑 smoke 训练；脚本能跑通只代表连通性，不是结论。
- 改动羽流模型后：重新生成羽流场图和传感器响应图，并跑 `analyze_plume_intermittency.py` 前后对比，确认 whiff/blank 结构仍合理。
- 改动评估指标后：保持两个 evaluate 脚本输出字段一致。
- 奖励阈值/系数是针对*当前*羽流浓度量级调的——若改了羽流量级或域随机化范围，需重新测量传感器读数分布并重标定奖励（步骤见 PROJECT_RECORD §17.6）。
