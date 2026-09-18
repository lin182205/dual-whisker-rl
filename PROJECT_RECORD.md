# 双触须主动嗅觉强化学习项目记录

本文档用于在后续开发过程中快速恢复项目目标、技术路线、已完成工作和下一步方向。开发前建议先阅读本文档，避免只围绕局部代码修改而偏离论文主线。

## 1. 项目目标

本项目面向气味源定位任务，目标是构建一个二维仿真环境，用强化学习研究“机器人移动控制”和“双触须主动嗅觉采样控制”的联合优化。

核心问题不是简单训练机器人沿浓度梯度爬坡，也不是只做固定双电子鼻对照，而是验证主动摆动的双触须采样结构是否能在间歇气味羽流中提供更有效的信息获取能力，从而提高气味源搜索效率。

论文主线可以概括为：

```text
双触须主动嗅觉采样 + 机器人移动联合控制
```

期望实验对比关系为：

```text
Joint RL: robot movement + whisker active sampling
>
Fixed-whisker RL: robot movement only, whiskers follow fixed scanning
>
Dual fixed electronic nose
>
Single fixed electronic nose
```

当前代码主要完成了前两个方法的最小闭环：联合控制 PPO 与固定触须 DQN。

## 2. 当前技术路线

当前阶段采用纯 Python 仿真，不涉及真实硬件。

仿真环境包含：

- 二维搜索区域；
- 固定气味源；
- 简化高斯气味羽流模型；
- 随机气味斑块和噪声，用于模拟羽流间歇性；
- 差速机器人运动模型；
- 左右双触须采样点；
- 一阶慢响应气体传感器；
- Gymnasium 强化学习接口；
- Stable-Baselines3 DQN 训练流程；
- 训练、评估、指标统计和可视化输出。

当前强化学习算法使用 Stable-Baselines3。固定触须 baseline 使用 DQN；联合控制方法使用 PPO，以支持 `MultiDiscrete([robot, left_whisker, right_whisker])` 动作空间。底层神经网络由 PyTorch 实现。

## 3. 代码结构

```text
rl/
├── configs/
│   └── default.yaml
├── dual_whisker_rl/
│   ├── envs/
│   │   ├── plume_env.py
│   │   ├── fixed_whisker_env.py
│   │   ├── plume_model.py
│   │   ├── robot_model.py
│   │   ├── whisker_model.py
│   │   └── sensor_model.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── agents/
├── scripts/
│   ├── random_rollout.py
│   ├── fixed_whisker_baseline_rollout.py
│   ├── train_fixed_whisker_dqn.py
│   ├── evaluate_fixed_whisker_dqn.py
│   ├── train_joint_dqn.py
│   └── evaluate_joint_dqn.py
├── requirements.txt
├── requirements-rl.txt
├── README.md
└── PROJECT_RECORD.md
```

`results/` 用于保存训练模型、日志、评估图像和动画，已加入 `.gitignore`，不应提交到版本控制。

## 4. 已完成工作

### 4.1 项目环境

已创建 Python 虚拟环境 `.venv`，并配置 VS Code 在新开集成终端时自动激活该环境。

依赖分为两类：

- `requirements.txt`：基础仿真依赖；
- `requirements-rl.txt`：强化学习训练依赖，包括 PyTorch、Stable-Baselines3、TensorBoard 等。

### 4.2 主环境 PlumeEnv

`PlumeEnv` 是主方法环境，支持机器人移动和触须采样联合控制。

动作空间为：

```text
MultiDiscrete([6, 10, 10])
```

机器人移动动作包括前进、左转、右转、原地左旋、原地右旋和停止。

左右触须动作相互独立，每根触须负责机器人对应侧 180 度半平面，并离散为 10 个扇区。左触须扇区中心角为 `+9, +27, ..., +171` 度；右触须扇区中心角为 `-9, -27, ..., -171` 度。

观测状态当前为 14 维，包含左右传感器读数、浓度差、浓度变化、左右 hit rate、触须角度、相对风向、机器人朝向和上一动作信息。

### 4.3 固定触须 baseline

`FixedWhiskerPlumeEnv` 是固定触须摆动 baseline。

该环境中智能体只控制机器人移动，触须按照 10 个扇区周期性扫描。动作空间降为：

```text
6 个机器人移动动作
```

这个 baseline 用于回答一个关键问题：

```text
如果触须只是固定摆动，RL 是否已经能完成基本找源？
```

只有在固定触须 baseline 之上，联合控制方法表现更好，才更能说明“主动控制触须采样”有额外价值。

### 4.4 传感器模型

当前传感器层面主要只有一阶动态响应模型：

```text
y_t = alpha * y_{t-1} + (1 - alpha) * concentration
```

默认配置：

```yaml
sensor_alpha: 0.95
```

这用于模拟气体传感器的慢响应和滞后现象。

当前没有实现复杂真实传感器特性，例如：

- 响应/恢复不对称；
- 饱和效应；
- 基线漂移；
- 温湿度补偿；
- 显式滤波器；
- 归一化；
- 延迟队列；
- 传感器标定曲线。

需要注意：`raw_left` 和 `raw_right` 是羽流模型在触须采样点处的原始浓度，主要用于记录和画图；`left` 和 `right` 是经过一阶动态响应后的传感器输出，也是智能体实际看到的主要浓度信息。

### 4.5 训练流程

已实现固定触须 DQN 与联合控制 PPO 两套训练脚本：

```powershell
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4
```

训练脚本默认使用随机 seed，以减少策略只适配单一初始化序列的风险。需要复现实验时显式传入：

```powershell
python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4 --seed 42
```

每次训练会在对应日志目录写入 `run_metadata.json`，记录实际 seed、环境 seed、配置和模型路径。

默认模型输出：

```text
results/models/fixed_whisker_dqn.zip
results/models/joint_ppo.zip
```

默认日志输出：

```text
results/logs/fixed_whisker_dqn/
results/logs/joint_ppo/
```

TensorBoard 训练曲线统一输出到：

```text
results/tensorboard/
```

启动方式：

```powershell
tensorboard --logdir results\tensorboard
```

训练过程中建议重点观察：

- `rollout/ep_rew_mean`：训练采样 episode 平均回报；
- `eval/mean_reward`：定期评估平均回报；
- PPO 的 `train/value_loss`、`train/policy_gradient_loss`、`train/entropy_loss`；
- DQN 的 TD loss 和探索率曲线。

已经做过 smoke training，证明训练流程可以运行。但 smoke training 只代表代码连通性测试，不能作为论文实验结论。

### 4.6 独立评估流程

评估逻辑已经从训练脚本中独立出来，统一放在：

```text
dual_whisker_rl/evaluation.py
```

评估脚本包括：

```powershell
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50
python scripts\evaluate_joint_ppo.py --episodes 50
```

当前评估指标包括：

- `success_rate`：搜索成功率；
- `mean_return`：平均回报；
- `mean_steps`：平均步数；
- `mean_final_distance`：最终距离源头的平均距离；
- `mean_path_length`：平均路径长度；
- `mean_odor_hits`：平均气味命中次数；
- `mean_reacquisition_time`：平均重新捕获气味耗时；
- `reacquisition_events`：重新捕获事件次数。

### 4.7 可视化

当前可视化能力包括：

- 气味羽流浓度场；
- 机器人轨迹图；
- 左右传感器响应曲线；
- 触须角度曲线；
- 评估轨迹 GIF 动画。

固定触须 DQN 评估已加入左右传感器响应曲线输出，图像格式参考 `sensor_response.png`，包含 raw 浓度和 sensor 输出。

默认输出位于：

```text
results/figures/
```

## 5. 当前重要理解

靠近气味源不一定意味着传感器浓度持续升高。

原因包括：

- 当前羽流模型主要在下风向有明显浓度；
- 机器人或触须可能越过气味源进入上风侧；
- 触须采样点不等于机器人中心点；
- 左右触须摆动时可能扫到羽流外侧；
- 间歇气味斑块会造成 raw 浓度突变；
- 一阶传感器会平滑响应，但不能保证读数单调增加。

因此，当前任务不是简单的浓度爬坡问题，而是带风向、间歇羽流、采样动作和传感器滞后的主动嗅觉搜索问题。

## 6. 当前阶段结论

项目已经具备最小实验闭环：

```text
仿真环境
baseline
主方法训练入口
统一评估
图像和动画可视化
```

但还没有完成正式实验对比，也还没有形成可用于论文的稳定结论。

当前最需要避免的是过早堆叠复杂模型。后续开发应优先保证：

1. baseline 公平；
2. 评估指标统一；
3. 多次实验可重复；
4. 结果能解释主动触须采样是否真的有收益。

## 7. 建议下一步

优先级从高到低：

1. 跑正式 fixed-whisker DQN 与 joint PPO 对比实验。
2. 增加 `scripts/compare_results.py`，汇总评估 JSON，输出对比表。
3. 增加多随机种子训练与评估，避免单次结果偶然。
4. 增加规则控制 baseline，例如沿风向/横风搜索策略。
5. 增加双固定电子鼻 baseline 和单电子鼻 baseline。
6. 视需要改进传感器模型，例如加入响应/恢复不对称、饱和、漂移或延迟。
7. 增加训练曲线可视化和论文图表自动生成。

建议先执行：

```powershell
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50

python scripts\train_joint_ppo.py --timesteps 50000 --n-envs 4
python scripts\evaluate_joint_ppo.py --episodes 50
```

之后再做结果汇总和对照实验扩展。

## 8. 开发注意事项

- 修改环境动力学、奖励函数或观测空间后，应重新跑 fixed-whisker 和 joint 两套 smoke training。
- 修改评估指标后，应保证两个评估脚本输出字段一致。
- 修改气味羽流模型后，应重新生成气味场图和传感器响应图，检查是否仍符合任务直觉。
- 不要把 `results/` 中的训练结果、模型和图像提交到 Git。
- README 用于快速启动，本文档用于保存项目路线和阶段性判断。

## 9. 硬件接入阶段计划

当前硬件条件已经明确：控制板使用 STM32，执行机构为两个舵机，传感器为两个 MQ-3 气体传感器，供电问题已经解决。下一阶段软件算法进度应暂时放慢，优先完成硬件最小闭环验证。该阶段目标不是直接在真实硬件上训练 PPO，而是先验证从 PC 到 STM32、再到舵机和 MQ-3 传感器的数据链路是否可靠。

硬件软件分工采用上下位机结构：

```text
PC Python 程序：上位机
STM32 C 程序：下位机固件
```

工具链确定为：

```text
STM32CubeMX
STM32CubeIDE
HAL 库
```

PC 端 Python 不直接产生 PWM，而是通过串口下发触须扇区命令。STM32 端 C 固件负责 UART 命令解析、TIM PWM 舵机控制、ADC 采样和串口数据返回。

第一版硬件闭环定义为：

```text
PC Python 程序
-> 串口发送左右触须扇区命令
-> STM32 接收命令
-> STM32 控制两个舵机转动到对应扇区角度
-> 等待舵机稳定
-> STM32 读取两个 MQ-3 ADC 值
-> STM32 通过串口返回数据
-> PC 保存 CSV/JSONL 并绘制传感器响应图
```

粗略跑通阶段使用最简单的串口文本协议即可。PC 端发送：

```text
STEP left_sector right_sector
```

例如：

```text
STEP 3 7
```

表示左触须转到第 3 扇区，右触须转到第 7 扇区。STM32 返回一行 JSON 文本：

```json
{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}
```

第一版返回字段只需要包含左右扇区和两个 MQ-3 原始 ADC 值。后续再扩展时间戳、舵机角度、滤波值、基线扣除值和归一化值。

STM32 端最小固件逻辑为：

```text
1. 初始化 UART，用于接收 Python 命令和返回数据；
2. 初始化 TIM PWM 两个通道，用于控制左右舵机；
3. 初始化 ADC 两个通道，用于读取左右 MQ-3；
4. 接收一行串口命令，例如 STEP 3 7；
5. 解析 left_sector 和 right_sector；
6. 将 sector 映射为舵机角度；
7. 将角度转换为 PWM 脉宽；
8. 设置左右舵机 PWM；
9. 等待 settle_ms；
10. 读取左右 MQ-3 ADC；
11. 串口返回一行 JSON 文本。
```

触须扇区到舵机角度的映射应与仿真保持一致。当前仿真中左右触须各 10 个扇区，对应 180 度半平面。硬件粗略阶段可以先采用：

```text
sector 0 -> 9 deg
sector 1 -> 27 deg
sector 2 -> 45 deg
...
sector 9 -> 171 deg
```

由于左右舵机实际安装方向可能不同，Python 配置中应保留 `invert` 和 `trim_deg` 字段，后续通过配置校准，不要在代码中硬编码左右舵机方向。

硬件阶段建议新增目录结构：

```text
rl/
├── configs/
│   └── hardware.yaml
├── dual_whisker_rl/
│   └── hardware/
│       ├── __init__.py
│       ├── serial_client.py
│       └── hardware_logger.py
├── hardware/
│   ├── firmware/
│   │   └── stm32_dual_whisker/
│   │       └── README.md
│   └── wiring/
│       └── wiring_notes.md
└── scripts/
    ├── hardware_smoke_test.py
    └── hardware_sector_scan.py
```

当前优先级最高的是硬件 smoke test。第一步在 STM32 固件中实现 `STEP left right` 命令解析、双舵机转动、双 MQ-3 ADC 采样和串口返回。第二步在 Python 端实现串口客户端，提供 `step(left_sector, right_sector)` 接口。第三步编写 `hardware_smoke_test.py`，循环发送 `STEP 0 0` 到 `STEP 9 9`，记录返回值并保存到 `results/hardware/`。

粗略跑通时不需要复杂滤波、标定或传感器建模。建议每次 `STEP` 后等待 300 到 500 ms，让舵机稳定；MQ-3 ADC 连续读取 10 次取平均；PC 端保存原始值并画出左右传感器曲线即可。等链路跑通后，再做 MQ-3 预热、基线漂移、响应/恢复时间、归一化和仿真传感器参数拟合。

硬件阶段暂时不接机器人底盘，也不直接进行真实硬件强化学习训练。建议先只验证：

```text
left_sector/right_sector
-> 舵机转动
-> MQ-3 响应
-> 数据回传
-> 曲线与日志输出
```

等触须硬件链路稳定后，再加载仿真中训练好的 PPO 模型，仅执行 PPO 输出中的左右触须扇区动作，观察策略选择是否能在硬件上产生可解释的 MQ-3 响应变化。机器人移动控制应放到更后阶段处理。

## 10. 硬件最小闭环当前进展

当前已经从纯仿真阶段进入硬件最小闭环搭建阶段。硬件平台确定为 STM32F103C8T6，执行机构为两个舵机，传感器为两个 MQ-3 气体传感器。硬件线路已经完成连接，供电问题已经解决，当前目标是先粗略跑通 `PC Python -> STM32 -> 舵机 -> MQ-3 -> PC` 的数据链路。

STM32CubeMX 已经配置并生成工程，源码目录为：

```text
hardware/firmware/stm32_dual_whisker/whisker/
```

当前采用 STM32CubeMX、STM32CubeIDE 和 HAL 库进行下位机开发。PC Python 程序作为上位机，只通过串口发送左右触须扇区命令；STM32 C 程序作为下位机固件，负责 UART 命令解析、TIM PWM 舵机控制、ADC 采样和串口返回数据。

当前硬件引脚分配为：

```text
左舵机 PWM 信号  -> PA0  / TIM2_CH1
右舵机 PWM 信号  -> PA1  / TIM2_CH2

左 MQ-3 AO       -> PA2  / ADC1_IN2
右 MQ-3 AO       -> PA3  / ADC1_IN3

USART1_TX        -> PA9
USART1_RX        -> PA10
```

需要持续注意：STM32F103C8T6 的 ADC 输入电压不能超过 3.3V。如果 MQ-3 模块 AO 输出可能高于 3.3V，需要通过分压或电平保护后再接入 PA2/PA3。

当前 `Core/Src/main.c` 已经写入最小闭环 HAL 代码。固件启动后会启动 TIM2 两路 PWM，并通过 USART1 等待 PC 端命令。PC 端发送：

```text
STEP left_sector right_sector
```

例如：

```text
STEP 3 7
```

STM32 解析命令后，将左右触须扇区映射为舵机角度，设置 TIM2_CH1 和 TIM2_CH2 的 PWM compare 值，等待舵机稳定，然后依次读取 PA2 和 PA3 上的两个 MQ-3 ADC 值，并通过串口返回：

```json
{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}
```

当前 TIM2 已调整为适合舵机控制的 50Hz PWM。配置为：

```text
Prescaler = 7
Period = 19999
```

在当前 8MHz 时钟配置下，计数单位约为 1us，PWM 周期约为 20ms。舵机角度到 PWM 脉宽的映射采用约 0.5ms 到 2.5ms 的常见舵机控制范围。

当前每次 `STEP` 的时间主要由舵机稳定等待和 ADC 多次采样组成。固件中设置：

```c
#define SERVO_SETTLE_MS 500
#define ADC_SAMPLE_COUNT 10
```

因此一次 `STEP` 大约包含 500ms 舵机稳定等待，加上左右两个 MQ-3 的顺序 ADC 采样时间。每个通道采样 10 次，每次间隔约 5ms，两个通道合计约 100ms。因此 STM32 端一次完整操作大约为 600ms。PC 端 smoke test 脚本默认还有 `--delay-s 0.2`，所以实际扇区切换间隔约为 0.8s。

左右舵机控制由 TIM2 的两个 PWM 通道并行输出。虽然代码中设置左右 compare 值有先后顺序，但时间差为微秒级，可以认为左右舵机同时控制。舵机机械运动阶段共同等待 500ms，不是左舵机等待 500ms 后右舵机再等待 500ms。

左右 MQ-3 当前采用顺序 ADC 采样。固件先读取左 MQ-3，再读取右 MQ-3，两个读数之间约有几十毫秒时间差。由于 MQ-3 响应速度本身较慢，这个时间差在 smoke test 阶段可以接受。后续如果需要更规范的同步采样，可以改为 ADC 扫描模式配合 DMA。

PC 端已经准备好硬件上位机脚本和串口客户端：

```text
dual_whisker_rl/hardware/serial_client.py
dual_whisker_rl/hardware/hardware_logger.py
scripts/hardware_smoke_test.py
scripts/hardware_sector_scan.py
```

硬件依赖记录在：

```text
requirements-hardware.txt
```

烧录 STM32 固件后，可以先运行：

```powershell
python scripts\hardware_smoke_test.py --port COM3
```

如果串口号不是 COM3，则替换为实际串口号。该脚本会依次发送 `STEP 0 0` 到 `STEP 9 9`，保存 CSV，并绘制左右 MQ-3 ADC 响应曲线。重复扫描可以运行：

```powershell
python scripts\hardware_sector_scan.py --port COM3 --cycles 5
```

当前尚未在本机完成 STM32 工程编译验证，因为命令行环境缺少 `cmake`。固件需要在 STM32CubeIDE 中编译、烧录和调试。Python 上位机脚本已经通过编译检查。

## 11. 硬件 smoke test 数据记录与当前理解

当前已经开始使用 Python 上位机脚本读取 STM32 返回的硬件数据。硬件 smoke test 的默认输出文件为：

```text
results/hardware/smoke_test.csv
```

该文件目前包含如下字段：

```text
step,t_pc_s,left_sector,right_sector,left_adc,right_adc
```

其中 `step` 表示上位机脚本记录的第几次操作；`left_sector` 和 `right_sector` 表示本次发送给 STM32 的左右触须扇区编号；`left_adc` 和 `right_adc` 表示 STM32 从左右 MQ-3 传感器 ADC 通道读取并返回的原始数值。当前 ADC 数值还没有做电压换算、基线扣除、归一化或气体浓度标定，因此现阶段主要用于判断传感器链路是否连通、左右传感器是否有响应、不同扇区下读数是否存在可观察变化。

`t_pc_s` 表示 PC 上位机记录该条样本时的时间戳。当前脚本使用的是 PC 系统绝对时间，因此数值会比较大，例如：

```csv
step,t_pc_s,left_sector,right_sector,left_adc,right_adc
0,1780247957.523811,0,0,805,2115
1,1780247958.355813,1,1,805,2058
```

这两条数据表示：第 0 步时左右触须都转到 0 号扇区，左 MQ-3 ADC 为 805，右 MQ-3 ADC 为 2115；第 1 步时左右触须都转到 1 号扇区，左 MQ-3 ADC 仍为 805，右 MQ-3 ADC 为 2058。两条样本之间的 PC 时间差约为：

```text
1780247958.355813 - 1780247957.523811 = 0.832002 s
```

这个间隔与当前固件和上位机脚本的设定基本一致：STM32 端一次 `STEP` 约包含 500ms 舵机稳定等待和约 100ms 左右传感器顺序 ADC 采样；PC 端脚本默认还有 `--delay-s 0.2` 的额外等待，因此相邻扇区命令之间大约为 0.8s。

当前 `scripts/hardware_smoke_test.py` 和 `scripts/hardware_sector_scan.py` 的核心控制流程非常接近，都是通过串口循环发送 `STEP sector sector`，等待 STM32 返回左右 ADC 数据，然后保存 CSV 并绘图。二者目前主要区别是实验用途、默认循环次数、输出文件名和绘图方式：

```text
hardware_smoke_test.py
用途：第一次验证串口、舵机、ADC、CSV 和绘图链路是否跑通。
默认 cycles：1
默认输出：results/hardware/smoke_test.csv 和 smoke_test.png
图像重点：按 step 展示左右 ADC 时间序列。

hardware_sector_scan.py
用途：重复扫描多个周期，用于观察不同扇区下 MQ-3 响应是否稳定、是否存在扇区相关差异。
默认 cycles：5
默认输出：results/hardware/sector_scan.csv 和 sector_scan.png
图像重点：按 sector 汇总多个周期中的左右 ADC 散点。
```

命令行参数使用 `argparse` 的命名参数形式，因此参数顺序没有强制要求。下面几种写法在含义上等价：

```powershell
python scripts\hardware_smoke_test.py --port COM3 --cycles 1 --delay-s 0.2
python scripts\hardware_smoke_test.py --cycles 1 --delay-s 0.2 --port COM3
python scripts\hardware_smoke_test.py --delay-s 0.2 --port COM3 --cycles 1
```

Windows 下串口名通常不区分大小写，因此 `COM3` 和 `com3` 一般都可以使用。为了与设备管理器和项目文档保持一致，建议统一写成大写 `COM3`、`COM4` 这类形式。

下一步建议先保持两个脚本分工不变。`hardware_smoke_test.py` 用于快速确认硬件链路是否正常，`hardware_sector_scan.py` 用于重复采样和初步分析扇区响应。如果后续发现两个脚本长期重复，可以再抽取公共扫描逻辑，或者合并为一个脚本并通过 `--mode smoke/sector-scan` 区分输出方式。

后续可以考虑对硬件数据记录做两个小改进。第一，将 `t_pc_s` 从绝对时间改为脚本启动后的相对时间，使 CSV 更便于阅读，例如第 0 步为 0.8s、第 1 步为 1.6s。第二，在 STM32 返回 JSON 中增加 `t_mcu_ms` 字段，使用 `HAL_GetTick()` 记录单片机端毫秒时间。这样后续可以同时分析 PC 端通信间隔和 STM32 端实际执行耗时，更容易定位串口等待、舵机稳定和 ADC 采样分别占用了多少时间。

## 12. 触须专用仿真环境与训练接口当前进展

在完成硬件最小链路后，项目开始补充一个单独用于触须训练的仿真环境。这个环境暂时不考虑机器人移动和障碍物，目标是先让左右触须在一个动态气体羽流中学习主动选择采样扇区。对应环境文件为：

```text
dual_whisker_rl/envs/whisker_only_env.py
```

该环境当前命名为 `WhiskerOnlyPuffEnv`。机器人本体在环境中固定不动，策略只输出左右触须扇区动作：

```text
MultiDiscrete([10, 10])
```

动作含义为：

```text
[left_sector, right_sector]
```

这与硬件串口协议保持一致，可以自然映射为：

```text
STEP left_sector right_sector
```

因此后续把仿真训练得到的触须策略接入 STM32 舵机触须时，不需要再重新设计动作接口，只需要让上位机加载策略模型，根据当前传感器观测输出左右扇区编号，再通过串口下发 `STEP` 命令。

触须专用环境的场地已经调整为 1m x 1m，坐标范围为：

```text
x, y in [-0.5, 0.5]
```

触须长度设置为 0.15m，即 15cm，用于贴近后续硬件触须尺度。环境中不再添加障碍物，便于先单独研究触须主动采样行为，而不是把问题混合成避障、移动控制和气体搜索的综合问题。

触须仿真已经加入舵机转动速度约束。当前按照常见舵机参数设置为：

```text
0.12 s / 60 deg
```

换算后角速度约为：

```text
500 deg/s
```

因此策略输出的左右扇区动作不再表示触须瞬间跳到目标角度，而是表示左右舵机的目标扇区。环境每一步会根据当前仿真时间步 `dt` 计算本步最多允许转过的角度，然后将左右触须从当前角度逐步转向目标角度。以触须专用环境当前默认 `dt = 0.2s` 为例，每一步最多转过：

```text
500 deg/s * 0.2 s = 100 deg
```

如果触须从 0 号扇区中心约 9 度转向 9 号扇区中心约 171 度，总角度差为 162 度，则至少需要两个环境步才能完全到位。这样训练时的采样点、观测角度和可视化触须位置都更接近真实舵机，而不是理想化的瞬时切换。

气体分布模型从原先较简单的静态高斯羽流，扩展为动态 puff 羽流。当前实现类为：

```text
DynamicPuffPlume
```

该模型会在场地外的上风侧放置气体源，使气体团随风进入 1m x 1m 训练区域。每个 episode 会随机化风向、风速、气源横向偏移和 puff 分布，因此训练时看到的气体场不是固定的一张图，而是具有随机性和动态变化的羽流。当前设计目标不是完全复现真实流体力学，而是构建一个比静态高斯场更接近真实气味羽流的训练场景：有方向性、有局部浓度斑块、有随时间变化的气体团，并且避免过于规则的几何中心点。

当前气体源默认位于场地外侧，羽流受风向影响进入全局。这样做的原因是，后续真实实验中气源通常不会恰好位于触须附近，触须采样更可能面对的是已经被气流输运和扩散后的宽羽流覆盖区。当前触须初始位置会优先采样在羽流覆盖区域附近，使策略训练时更容易获得有效气味信号，而不是长时间停留在全零或接近零的观测中。

触须专用环境继续沿用已有传感器模型：

```text
dual_whisker_rl/envs/sensor_model.py
```

也就是一阶慢响应气体传感器模型。环境 step 的基本逻辑为：

```text
1. PPO 策略输出 left_sector 和 right_sector
2. 环境推进一次动态 puff 气体场
3. 左右触须以舵机角速度限制转向目标扇区
4. 在新的左右触须端点位置读取瞬时气体浓度 raw_left/raw_right
5. 经过一阶慢响应模型得到 left/right 传感器读数
6. 根据气味存在、气味增强趋势、左右差异和时间惩罚计算 reward
7. 返回下一帧观测
```

当前基础观测维度为 12 维，包含左右传感器读数、左右差值、左右读数变化、近期 hit rate、左右触须角度、相对风向和到气源的距离等信息。为了让策略看到 MQ-3 这类慢响应传感器的短期变化趋势，训练脚本中新增了历史观测堆叠包装器 `ObservationHistoryWrapper`，默认把最近 6 帧观测拼接为一个更长的状态向量。因此 PPO 实际看到的不只是单帧读数，而是一小段时间窗口内的传感器变化。

触须专用 PPO 训练脚本为：

```text
scripts/train_whisker_only_ppo.py
```

推荐基础训练命令为：

```powershell
python scripts\train_whisker_only_ppo.py --timesteps 50000 --n-envs 4 --history-length 6
```

其中 `--n-envs` 表示并行创建多个独立环境，每个环境拥有不同随机 seed 和不同动态气体羽流，用于提高策略泛化性；`--history-length` 表示堆叠多少帧历史观测。训练结束后默认保存模型到：

```text
results/models/whisker_only_ppo.zip
```

训练脚本会写入 TensorBoard 日志和 `run_metadata.json`，记录实际 seed、环境 seed、历史长度、动作空间、模型路径等信息，方便后续复现实验。

触须专用环境可视化脚本为：

```text
scripts/visualize_whisker_only_env.py
```

该脚本会生成一张静态图和一张 GIF 动态图：

```text
results/figures/whisker_only_env.png
results/figures/whisker_only_env.gif
```

当前可视化风格参考了已有 TD3 气体环境的 plume demo：背景为浅色，低浓度气体接近深色，高浓度逐渐过渡到黄色，并带有 colorbar。浓度图采用平滑插值，不再显示浓度层之间的黑色线条。图中会显示动态气体场、机器人位置、左右触须、触须端点、上风侧气源方向提示、风速风向、传感器读数、扇区编号和累计 reward。

动画导出当前支持以下关键参数：

```powershell
python scripts\visualize_whisker_only_env.py --steps 120 --realtime-speed 2 --interpolation-frames 2
```

其中 `--realtime-speed` 可以设置为小数或大于 1 的数。小于 1 表示慢放，大于 1 表示加速播放。`--interpolation-frames` 表示相邻仿真关键帧之间插入多少过渡帧，用于让 GIF 更丝滑。需要注意的是，提高插帧数会增加 GIF 文件大小。

之前发现一个动画问题：插帧时如果直接对触须端点坐标做线性插值，端点会沿直线运动，视觉上会导致触须长度在转动过程中变短。当前已经修复为“角度插值 + 固定长度重建端点”的方式。也就是说，动画插帧时先根据前后两帧的触须角度沿最短角度方向插值，再使用固定 `whisker_length` 重新计算端点位置，使触须端点沿圆弧运动，视觉长度始终保持 15cm。

当前触须专用环境与硬件链路之间的关系可以理解为：

```text
仿真环境：训练触须如何根据历史气味读数选择左右扇区
硬件链路：验证 PC 能否把左右扇区命令下发给 STM32，并读回 MQ-3 响应
后续部署：用训练好的 PPO 策略替代人工扇区扫描，由策略实时输出 STEP left right
```

因此短期内不建议急着把完整机器人移动控制接入硬件。更稳妥的路线是先完成“固定位置双触须主动采样”的 sim-to-real 小闭环：在真实气体环境中固定触须基座位置，采集 MQ-3 历史读数，PPO 策略输出左右扇区，STM32 控制舵机转动并返回新读数。等这个链路能稳定运行并表现出可解释的主动采样倾向后，再考虑把机器人移动控制重新加入。

下一步建议围绕触须专用环境做三件事。第一，跑较短 PPO 训练并观察 TensorBoard 中 reward 是否稳定上升，同时查看策略扇区选择是否明显偏向高信息区域。第二，将仿真环境中的传感器响应参数与真实 MQ-3 基线测试数据做初步拟合，尤其是慢响应时间常数和噪声范围。第三，编写一个硬件策略推理脚本，加载 `whisker_only_ppo.zip`，用真实 MQ-3 读数构造历史观测，并把策略输出转换为 `STEP left_sector right_sector` 串口命令。

## 13. 传感器预处理模块与下一步实用计划

在真实硬件测试中已经观察到 MQ-3 传感器存在明显问题：即使经过充分预热，在洁净空气中采集时，ADC 读数仍会随时间缓慢下降，且每次重新上电后的基线不一致。这说明当前不能把 MQ-3 原始 ADC 直接当作稳定的绝对浓度输入，也不适合直接拿 `left_adc` 和 `right_adc` 的绝对值喂给 PPO。

为了解决真实传感器输入不稳定的问题，已经新增硬件传感器预处理模块：

```text
dual_whisker_rl/hardware/sensor_preprocess.py
```

该模块的目标不是把 MQ-3 标定成精确浓度计，而是把漂移、噪声、慢响应明显的原始 ADC 转换成强化学习更容易利用的相对特征。当前包含：

```text
OnlineGasPreprocessor    单个 MQ-3 在线预处理器
DualGasPreprocessor      左右 MQ-3 成对预处理器
SensorPreprocessConfig   预处理参数配置
SensorFeatures           单传感器特征结果
DualSensorFeatures       左右传感器组合特征结果
```

单个传感器当前提取的核心特征包括：

```text
norm_signal           相对基线变化，表示 raw_adc - baseline
norm_smooth           EMA 平滑后的相对信号
norm_trend            短时间窗口内的变化趋势
norm_estimated_input  基于一阶响应模型反推的当前刺激估计
```

左右两个传感器还会额外计算差分特征：

```text
norm_signal_diff
norm_smooth_diff
norm_trend_diff
norm_estimated_input_diff
```

这些差分特征用于表示左右触须哪一侧气味响应更强、哪一侧趋势更明显，从而为主动触须采样提供方向性信息。后续 PPO 不应主要依赖原始 ADC 绝对值，而应更多依赖相对基线变化、平滑响应、趋势、左右差分和历史观测。

当前传感器预处理的意义可以概括为：

```text
raw ADC
-> baseline subtraction
-> EMA smoothing
-> trend / response feature
-> normalization
-> PPO observation
```

也就是说，它是仿真训练和真实硬件之间的观测适配层。没有这个层，仿真中训练出的策略很可能因为真实 MQ-3 的基线漂移、噪声和每次上电差异而失效。

完成传感器预处理模块后，下一步最实用的工作不是立刻增加更复杂的神经网络，也不是马上在硬件上部署 PPO，而是先验证预处理结果是否真的把真实传感器数据变成了可解释、可训练的状态。

建议下一步按如下顺序推进。

第一步，编写硬件基线分析脚本，用已有 `results/hardware/live_plot.csv` 检查预处理效果。建议新增：

```text
scripts/analyze_hardware_baseline.py
```

该脚本读取洁净空气采集数据，输出左右传感器的均值、标准差、最大值、最小值、漂移斜率、前后窗口差值，并使用 `DualGasPreprocessor` 生成预处理后的曲线图。图中建议同时画出：

```text
raw_adc
baseline
signal = raw_adc - baseline
smooth
trend
estimated_input
```

这一步的目标是回答一个实际问题：当前预处理参数能不能把洁净空气中的缓慢漂移压到较小范围，同时不把真实气体刺激完全吃掉。

第二步，把 `DualGasPreprocessor` 接入实时绘图脚本：

```text
scripts/hardware_live_plot.py
```

实时图不应只显示 `left_adc/right_adc`，还应至少显示：

```text
left/right raw ADC
left/right baseline-subtracted signal
left/right smooth
left-right smooth diff
```

这样在做真实气体实验时，可以直接观察当前触须采样是否带来了可解释的相对响应，而不是只能看漂移明显的原始 ADC。

第三步，重新定义“硬件可部署 PPO 观测”。当前触须仿真环境中仍有一些仿真里容易获得、但硬件上不一定能直接获得的信息，例如真实风向、气源距离等。后续如果目标是把触须 PPO 迁移到 STM32 舵机触须，观测应逐步改成硬件可获得特征，例如：

```text
left_norm_signal
right_norm_signal
left_norm_smooth
right_norm_smooth
left_norm_trend
right_norm_trend
left_right_smooth_diff
left_right_trend_diff
left_angle_norm
right_angle_norm
left_sector_norm
right_sector_norm
```

然后继续使用历史拼接：

```text
history_length = 10 / 20 / 30
```

其中当前触须仿真 `dt = 0.2s` 时，`history_length = 20` 大约表示 4 秒历史。考虑到 MQ-3 慢响应特性，后续建议优先尝试 `history_length = 20`，并与 `history_length = 1`、`10`、`30` 做对比。

第四步，在仿真里引入与硬件预处理一致的观测形式。理想情况下，训练 PPO 时看到的状态和硬件部署时构造的状态应尽量一致。也就是说，仿真环境里也应通过类似的传感器预处理逻辑输出相对信号、平滑信号、趋势和左右差分，而不是让策略依赖仿真特有的真实浓度或气源信息。

第五步，再进行触须专用 PPO 训练和硬件推理脚本开发。建议顺序为：

```text
1. 先用硬件数据验证预处理参数
2. 再把预处理接入实时绘图
3. 然后调整仿真 PPO 观测，使其更接近硬件输入
4. 重新训练 whisker-only PPO
5. 最后编写 hardware_whisker_policy_rollout.py 做真实舵机触须策略推理
```

这个阶段最务实的短期目标可以定义为：

```text
让真实 MQ-3 数据经过预处理后，形成稳定、可解释、可画图、可拼接历史的 PPO 输入特征。
```

只有这个目标完成后，继续训练 PPO 或做 sim-to-real 部署才更有意义。否则策略可能只是在仿真里学得很好，但到了真实硬件上面对漂移和基线不一致的 ADC 时无法稳定工作。

## 14. 主动触须有效性问题与短期验证计划

当前对项目目标有了更清晰的修正：双触须主动探测不一定必须实现“左触须指向高浓度区域、右触须指向低浓度区域，并最大化绝对左右差分”这一理想形式。更现实也更适合当前 MQ-3 硬件条件的目标是：主动触须能否比固定摆放的两个传感器提供更有效、更高信息密度的局部气味观测。

也就是说，触须主动探测的核心价值可以重新表述为：

```text
哪里更可能有气味
哪里能产生响应变化
哪里能制造左右不对称
哪里值得继续采样
```

只要主动触须能在这些方面为机器人搜索提供帮助，就已经有研究意义，不必一开始就要求它形成精确的实时空间浓度梯度。

需要承认的是，MQ-3 的慢响应、基线漂移、每次上电基线不一致，会削弱“实时左右空间差分”的可靠性。当前更合理的判断是：

```text
用 MQ-3 做精确实时气味梯度测量：难度较高。
用 MQ-3 验证主动触须采样能提高气味信息获取：仍然可行。
```

主动触须相比固定双传感器的潜在优势在于：固定传感器只能采样两个固定位置，而主动触须可以在机器人不移动的情况下扩大局部采样范围。固定传感器看到的是：

```text
左固定点 + 右固定点
```

主动触须可以看到：

```text
左侧多个扇区 + 右侧多个扇区
```

因此在羽流边缘、间歇气味斑块附近、固定传感器没有刚好落在高信息区域时，主动触须有可能通过选择不同扇区获得更高的气味命中率、更明显的趋势变化或更强的左右不对称。

但主动触须也可能没有提升，甚至因为传感器慢响应而造成信息混叠。尤其是如果触须转动太快，当前读数可能仍然主要反映前几个扇区的历史，而不是当前扇区。因此后续实验中必须注意：

```text
不要只看 raw ADC
不要让触须无意义快速乱扫
必须使用历史观测
必须使用预处理后的 smooth/trend/diff 特征
必须与固定传感器 baseline 做公平比较
```

短期最实用的验证计划应先从“触须信息获取能力”入手，而不是直接比较完整机器人找源成功率。建议先固定机器人位置、固定气源或气流条件，对比以下几类触须策略：

```text
1. Fixed sensors
   左右传感器固定在某个角度不动。

2. Periodic scan
   左右触须按固定周期扫描 10 个扇区。

3. Random scan
   左右触须随机选择扇区。

4. Active whisker
   PPO 或启发式策略根据历史读数选择扇区。
```

其中 Fixed sensors 不应只设置一个角度，否则 baseline 不公平。建议至少设置多个固定角度：

```text
fixed_30deg
fixed_60deg
fixed_90deg
fixed_120deg
fixed_150deg
```

如果主动触须只能优于某个很差的固定角度，结论较弱；如果能优于最佳固定角度或接近最佳固定角度，同时具备更强的重新捕获能力，则结论会更有说服力。

建议使用以下信息获取指标，而不是只看最终找源结果：

```text
气味命中率：
max(left_smooth, right_smooth) > threshold 的比例。

平均有效响应：
mean(max(left_smooth, right_smooth))。

响应变化能力：
mean(abs(left_trend)) + mean(abs(right_trend))。

左右不对称强度：
mean(abs(left_smooth - right_smooth))。

重新捕获能力：
从低响应状态到再次超过阈值所需时间。

高信息采样比例：
采样后产生正 trend 或高 contrast 的步数比例。
```

这些指标更符合当前目标：验证主动触须是否提高了局部气味信息获取，而不是强迫它直接输出精确气源方向。

接下来项目短期目标应调整为：

```text
目标 A：验证预处理后的 MQ-3 特征是否稳定、可解释。
目标 B：比较固定传感器、周期扫描、随机扫描在真实硬件上的信息获取指标。
目标 C：如果周期扫描已经优于固定传感器，再训练或部署主动触须 PPO。
目标 D：如果周期扫描都没有优于固定传感器，则优先调整硬件采样节奏、停留时间、传感器位置和预处理参数，而不是盲目训练 PPO。
```

这个顺序的原因是：如果最简单的周期扫描都无法比固定传感器带来更多有效气味信息，那么复杂 PPO 主动控制很可能也很难在真实 MQ-3 硬件上表现出稳定收益。反过来，如果周期扫描已经显示出更高命中率、更强趋势或更明显左右差分，那么说明主动触须确实有潜力，后续训练 PPO 才更有意义。

因此下一阶段最建议新增的脚本是：

```text
scripts/compare_hardware_sampling_modes.py
```

该脚本可以统一运行或分析以下模式：

```text
fixed_angle
periodic_scan
random_scan
```

输出统一 CSV 和指标 JSON，后续再把 `active_policy` 模式接进去。这样可以先用真实硬件验证“主动/扫描触须是否比固定传感器更有信息”，再决定是否继续投入 PPO 训练和部署。

## 15. 阶段一完成：硬件基线分析与传感器预处理验证

已经新增并跑通硬件基线分析脚本：

```text
scripts/analyze_hardware_baseline.py
```

该脚本属于短期计划的阶段一（传感器数据质量验证）。它不训练 PPO，只用已采集的真实 MQ-3 CSV 离线验证 `dual_whisker_rl/hardware/sensor_preprocess.py` 是否能把漂移、噪声、慢响应明显的原始 ADC 转换成稳定可解释的特征。脚本完成三件事：统计 raw ADC 质量（均值、标准差、最值、漂移斜率、首尾窗口差、去趋势噪声）；在完整连续数据流上运行 `DualGasPreprocessor` 得到 baseline/signal/smooth/trend/estimated_input 曲线；输出 JSON、PNG、summary.md 并给出可用性判定。

脚本默认输出（带 `--label` 后缀区分不同分析段）：

```text
results/hardware/baseline_analysis<_label>.json
results/hardware/baseline_preprocess<_label>.png
results/hardware/baseline_summary<_label>.md
```

### 15.1 关键设计：连续预处理 + 分析窗口

预处理器是有状态的在线模块（baseline 和 EMA 不断累积），因此脚本始终在**完整连续数据流**上运行预处理，以匹配真实硬件部署时的输入。`--t-start` 和 `--t-end` 只是框出一个**分析窗口**，用于计算分段统计、判定和绘图，并不截断预处理输入。这样做的原因是：洁净段和气体段回答两个相反的问题（漂移抑制 vs 响应保留），而同一套参数必须同时满足这两个互相拉扯的要求；混在一起算总均值会把这个矛盾平均掉、看不见。`--segment-kind clean/gas/mixed` 决定判定逻辑：clean 期望特征接近 0，gas 期望有明显响应且未饱和。

### 15.2 第一份数据的教训：饱和与定标

第一份 `live_plot.csv` 是“洁净空气 + 一次强气体刺激直接打饱和 + 慢恢复”的混合记录，ADC 顶到 4095（12-bit 上限）。这暴露两个真实问题：一是气源太近/太浓导致传感器饱和，饱和状态下左右强弱和梯度信息在传感器端就已丢失，调任何软件参数都无法恢复，必须退远气源/降低浓度；二是默认 `scale=200` 偏小约一个数量级，真实响应约 3000 ADC 会被 clip 截断。同时也暴露了混合分析的误导性：混着算左通道漂移显示 2702 ADC，其实绝大部分是气体和饱和贡献，真实洁净基线漂移被完全淹没。

### 15.3 验收结论（第二份不饱和数据）

重新采集一段不饱和的连续数据（洁净基线 0~63s → 气体进入 76s → 多次气体团峰值 2000~3900 → 长慢恢复尾巴至 810s）后，用统一 `scale=1500` 分两段验证，结果如下：

```text
洁净段 (0~60s):  status=PASS
  左右基线漂移 ≈ -8.7 / +1.6 ADC（60s），去趋势噪声 std ≈ 1.4 / 2.8 ADC
  max|norm_smooth| ≈ 0.01, max|norm_trend| ≈ 0.00

气体段 (76s+):   status=RESPONSE_OK, saturated=False
  max ADC ≈ 2800 / 3979（未撞 4095）
  max|norm_smooth| ≈ 2.12, max|norm_trend| ≈ 0.16
```

核心结论是：洁净空气里预处理后特征基本为平（约 0.01），气体来时清晰抬升到约 2.12，**“无气味”与“有气味”信噪对比拉开约 200 倍**，且安全落在 clip=5 以内仍有余量，全程未饱和。这说明真实 MQ-3 数据经过预处理后已经是稳定、可解释、可拼接历史的特征，阶段一正式通过。

### 15.4 标定的预处理参数

当前这套硬件标定后的可用预处理参数为：

```text
scale          = 1500   （这套硬件的合适默认值，第一份数据用 200 偏小）
baseline_tau_s = 180
smooth_tau_s   = 2.0
response_tau_s = 2.0
trend_window   = 8
dt_s           ≈ 0.63   （从 CSV 时间戳自动推算，不手动指定）
```

后续采样模式对比和触须 PPO 应统一使用这套预处理参数。需要注意：这套 `scale=1500` 是针对当前气源距离/浓度/传感器布置标定的；若实验布置改变，应重新用本脚本的 gas 段验证响应幅度并重新定标。

### 15.5 采集规范（避免再次饱和）

后续采集气体数据时应保证：气体峰值 ADC 落在约 1500~3000、不要碰 4095；录一段完整连续过程（洁净基线 → 引入气源 → 升高维持 → 撤掉气源 → 慢恢复尾巴），中途不断电、不重新上电，保持预处理连续累积；触须先固定不动（如 `STEP 5 5`），先单独看传感器响应，不要同时引入扇区变化变量。

### 15.6 阶段一之后的下一步

阶段一已通过，按可行性文档的实验决策树，下一步进入阶段二：实现 `scripts/compare_hardware_sampling_modes.py`，在真实硬件上比较 fixed_angle、periodic_scan、random_scan 三种采样方式的信息获取指标，统一复用本阶段标定的预处理参数（`scale=1500` 等）。

### 15.7 对比图改为双 Y 轴叠加，及恢复尾巴被 baseline 吃掉的观察

脚本现在同时输出**两种图**，两者都保留：

```text
baseline_preprocess<_label>.png  分面板图（上一版风格）：raw/baseline、signal、smooth+est_input、trend+diff 各占一行，看每个特征自身细节。
baseline_overlay<_label>.png     双 Y 轴叠加对比图：左右传感器各一行，左轴(蓝)画绝对量 raw ADC 与 baseline，右轴(红)画预处理后相对量 signal 与 smooth，共享同一时间轴，直接对比原始与预处理曲线。
```

叠加图能直观看到 smooth 与 raw 形状完全吻合，只是去掉了基线偏置和噪声。新增 `--overlay-path` 可单独指定叠加图路径，`--figure-path` 指定分面板图路径。

这张叠加图还暴露出一个之前分开看不易发现的现象：在很长的慢恢复尾巴段（示例数据 t≈450~550s 起），baseline 自动更新逻辑会重新启动并向上追 raw。原因是恢复尾巴后期 smooth/trend 已降到 `_can_update_baseline` 的门限以下，预处理判定“差不多回到洁净空气”就开始缓慢更新基线，但传感器实际仍在缓慢恢复。结果是 `signal = raw - baseline` 中的 baseline 被抬高，**长时间残留气味会被 baseline 部分吃掉**，红色 smooth 比 raw 提前掉向 0。

这对阶段一结论没有影响（洁净段平、气体段响应明显且未饱和的判定不变）。但如果后续要让策略利用“长时间残留气味”这类慢信息，需要把 `baseline_tau_s` 调更大，或在确认进入气体刺激阶段后用脚本已有的 `--freeze-baseline` 冻结基线，避免真实响应被缓慢更新的基线抵消。

## 16. 阶段二框架：硬件采样模式对比脚本

已经搭好阶段二脚本框架（尚未接硬件实采）：

```text
scripts/compare_hardware_sampling_modes.py
```

该脚本用于在真实硬件上比较三类触须采样方式的“信息获取能力”，回答整个课题最关键的问题：扫描触须是否真的比固定角度传感器获得更多有效气味信息。它复用阶段一标定的预处理参数（默认 `scale=1500`），把原始 ADC 转成 `norm_smooth/norm_trend/smooth_diff` 等相对特征，再据此计算统一指标。

### 16.1 支持的模式与运行方式

支持三种采样模式：

```text
fixed_angle    左右触须固定在某扇区不动（--fixed-sector）
periodic_scan  往返三角波周期扫描 0->9->0，每扇区停留 --dwell-steps
random_scan    随机扫描，--random-mode any(任意扇区) / neighbor(只动±1，更贴合舵机)
```

左右触须采用对称扇区（与现有 smoke_test/sector_scan 约定一致），但左右物理基座位置不同，仍是两个采样点。

提供两种运行方式：

```text
实采：连 STM32 按模式下发 STEP 并采集（需要硬件）
回放：--replay-csv 读已有 CSV，只复算特征/指标/图（离线，无需硬件）
```

回放模式既用于离线测试指标与绘图逻辑，也可对历史记录重新分析。当前已用固定扇区的 `live_plot.csv` 做回放自测，脚本端到端跑通：固定扇区下 `sector_coverage=0.10`、`sector_entropy=0`，符合预期。

### 16.2 统一信息获取指标

每次运行输出 `metrics.json`，统一指标包括：

```text
odor_hit_rate          max(left_smooth,right_smooth) 超阈值比例
mean_max_smooth        平均有效气味响应
mean_abs_smooth_diff   左右不对称强度
mean_abs_trend_sum     响应变化能力
positive_trend_rate    正趋势比例
high_information_rate  高响应/高差分/高趋势三者之一满足的比例
reacquisition_events   从丢失到重新捕获的事件数
mean_reacquisition_time_s 平均重新捕获耗时
sector_entropy_bits    访问扇区分布熵
sector_coverage        访问过的扇区占比
```

阈值可调：`--hit-threshold`（默认 0.2，对应 norm_smooth）、`--diff-threshold`、`--trend-threshold`。

### 16.3 输出结构

每次运行在 `results/hardware/sampling_modes/<run_id>/` 下保存：

```text
raw.csv       step,t_rel_s,mode,left/right_sector,left/right_adc
features.csv  左右 smooth/trend/norm 特征与差分
metrics.json  统一指标 + 预处理配置 + 阈值 + 环境备注
summary.md    指标摘要表
plots.png     max_smooth(含阈值线) / sector 时间序列 / 左右 smooth diff
```

`run_id` 默认按时间和模式生成，可用 `--run-id` 覆盖；`--source-note` 记录气源/风扇/距离等环境信息。

### 16.4 下一步：接硬件实采三种模式

脚本框架已就绪，剩下需要接 STM32 实采。建议每种模式各采一段（保持不饱和、同一天同一环境、记录 `--source-note`），固定角度至少测多个代表性扇区（如 sector 1/3/5/7/9），不要只测一个，避免 baseline 不公平。采完后按决策树判断：周期扫描若在 `odor_hit_rate`、`mean_abs_trend_sum` 或 `mean_reacquisition_time_s` 上优于多数固定角度，说明主动/扫描触须有潜力，可进入 PPO；若都不如固定角度，先回去调采样节奏、停留时间、气源布置，而不是急着训 PPO。后续再把 `random_neighbor`、`heuristic_active`、`ppo_active` 模式接入对比。

## 17. 仿真羽流与传感器真实度改造（间歇性 + 非对称传感器 + 域随机化）

针对“自建 2D 仿真与现实差距大、气体扩散不够真实”的担忧，对 `WhiskerOnlyPuffEnv` 的气味场和传感器做了一轮真实度改造。核心判断是：决定 sim-to-real 成败的不是绝对浓度算得准，而是**羽流间歇性（whiff/blank 结构）**和**观测空间对齐**。改造坚持“先量基准、再改、再用同一指标验证”的流程，避免凭感觉调参。

### 17.1 新增间歇性诊断脚本

```text
scripts/analyze_plume_intermittency.py
```

不训练、不改环境，把 `DynamicPuffPlume` 跑一段长时间，在气源下风向固定距离、沿横风向取一排采样点，记录瞬时浓度时间序列，计算并绘制：intermittency factor（高于阈值的时间占比）、whiff 时长分布、blank 时长分布、浓度时间序列。输出：

```text
results/figures/plume_intermittency_<label>.png
results/figures/plume_intermittency_<label>.json
```

用途是把“感觉有 gap”变成可测量、可对比、论文可引用的证据。改进羽流前后用同一脚本复跑对比即可。

### 17.2 改造前基准：羽流是连续饱和气幕，零间歇

改造前的诊断基准非常清晰：中心线和边缘（横风 0.0~0.3m）的 intermittency factor 全部为 **1.00**，没有任何 blank，平均浓度高达 5~17（而 hit_threshold=0.08）。即旧羽流在整个场地是一张连续饱和气幕，传感器几乎永远“在羽流内”。这种场景下主动采样没有可利用的结构，正是 2D 仿真“不真实”的根源。

### 17.3 羽流改造（A：相干湍流 + meander；B：filament 密度）

`DynamicPuffPlume` 的主要改动：

```text
1. 小尺度湍流：把每步独立白噪声抖动，改为 per-puff OU 速度扰动（时间相关，AR(1)）。
   同一 filament 的扰动在时间上连贯 → 形成蜿蜒成缕的相干气缕，而非各自乱抖。
2. 大尺度蜿蜒：平均风向走 OU 过程（均值回归到 base），两种风模式都生效。
   固定模式 = 固定平均风 + 湍流摆动，因此固定诊断下羽流仍是间歇的。
   plume strand 整体摆动是固定点产生 whiff/blank 的主要来源。
3. filament 密度/寿命：调成更细、更短寿命、适度密集，避免老 puff 膨胀叠加成连续气幕。
```

关键新增参数（`DynamicPuffPlume.__init__`）：

```text
wind_dir_meander_theta / wind_dir_meander_sigma   平均风向 OU（蜿蜒幅度）
turbulence_vel_relax                               per-puff 速度 OU 相关系数
turbulence_vel_std_downwind / _crosswind           per-puff 湍流速度强度（m/s）
puff_init_sigma_*, decay_rate, max_puff_age, ...   filament 尺度/寿命
```

改造后诊断结果（与基准对比）：

```text
            改造前                       改造后
中心线:     interm=1.00, 无 blank,      interm≈0.74, whiff≈2.5s / blank≈0.9s,
            mean_c≈17                    mean_c≈0.22
边缘0.3m:   interm=1.00, 无 blank,      interm≈0.68, whiff≈2.3s / blank≈1.1s,
            mean_c≈5.3                   mean_c≈0.20
```

从“全程饱和、无间歇”变为“秒级一缕一缕经过、有中心→边缘梯度”，正是真实间歇羽流的结构。可视化 `results/figures/whisker_only_env.png` 也从平滑团块变为可见丝状气缕。

### 17.4 非对称气体传感器（E）

```text
dual_whisker_rl/envs/sensor_model.py -> AsymmetricGasSensor
```

相比对称一阶模型，加入三个真实 MQ-3 特性：

```text
1. 响应/恢复非对称：上升用较快 response_tau，下降用较慢 recovery_tau
   （MQ-3 吸附快、解吸慢）。tau 经 dt 换算为 alpha = exp(-dt/tau)。
2. 基线漂移：输出叠加缓慢 OU 漂移，模拟上电后基线缓慢变化。
3. 测量噪声：输出叠加高斯噪声。
```

`WhiskerOnlyPuffEnv` 默认改用该模型（可用 `sensor_model='first_order'` 回退）。默认 `response_tau=0.6`、`recovery_tau=3.0`，已验证“快升慢降”（5 步升到 0.81，再 5 步只回落到 0.58）。把噪声/漂移放进仿真传感器，是为了让仿真观测与真实 MQ-3 的“脏”数据同分布，使预处理层和策略在 sim-to-real 时不至失效。

### 17.5 域随机化（D）

每个 episode 在标称值附近随机化羽流物理（释放率、扩散、衰减、湍流强度、meander 幅度、puff 寿命）和传感器特性（响应/恢复时间、噪声、基线漂移）。默认关闭（便于复现和诊断），训练时打开：

```powershell
python scripts\train_whisker_only_ppo.py --timesteps 50000 --n-envs 4 --history-length 6 --domain-randomization
```

实现：`DynamicPuffPlume.domain_randomization` + `_randomize_params()`；环境侧 `WhiskerOnlyPuffEnv._randomize_sensors()`，并在每次 `reset` 重建传感器以同步当前 rng 和随机参数。DR 是跨 sim-to-real gap 的真正杠杆，比单一“逼真场景”更鲁棒。

### 17.6 奖励按新浓度量级重标定（已完成）

旧奖励阈值/系数是按改造前饱和羽流（mean_c≈15）调的。在改造后羽流上量出触须端点处传感器读数分布：`max(left,right)` 典型 0.3~1.3，≥0.08 占 99.9%、≥0.30 占 93.5%。这说明**二值“in-plume / strong”bonus 在新羽流下几乎一直为真、不再区分动作好坏**；真正依赖动作的可学习信号是**趋势（trend）**和**左右对比（contrast）**。

因此重标定思路不是简单上移阈值，而是**把奖励重心移到连续、依赖动作的项（幅值 + 趋势 + 对比），并把近乎常驻的二值 bonus 权重调小**（与可行性文档 §10.5 的 reward 设计一致）。重标定后默认值：

```text
hit_threshold        = 0.08   仅用于观测 hit-rate 和初始位姿采样
strong_threshold     = 0.60   强响应 bonus 门限（env 级，已与 plume 解耦）
odor_hit_reward      = 0.08   presence 比例项，clip 到 presence_clip
presence_clip        = 1.5
trend_reward_scale   = 1.0    trend_clip = 0.12（旧为 4.0 / 0.04，新量级下会饱和）
contrast_reward_scale= 0.15   旧为 0.04，太小
tracking_bonus       = 0.05   旧为 0.12（近常驻，调小）
strong_bonus         = 0.05   旧为 0.10
time_penalty         = 0.04
```

随机策略下各项均值：trend 0.019(±0.045)、presence 0.061(±0.028)、contrast 0.049(±0.037)、track 0.050(±0.002)、strong 0.032。可学习项方差显著、常驻项方差≈0，平衡合理。

短训练验证（30k steps，仅 smoke）：训练后 reward 61.2 ≥ 随机 57.4；左右扇区选择熵 2.90 / 2.38 bits（满熵 3.32），覆盖 0.9~1.0，**未塌缩到单一扇区**，开始形成偏好。正式结论仍需 200k+ 多 seed 训练。注意：若改了 plume 量级或 DR 范围，应重跑本节分布测量并重标定。

### 17.7 本轮新增/改动文件小结

```text
新增  scripts/analyze_plume_intermittency.py     羽流间歇性诊断
改动  dual_whisker_rl/envs/sensor_model.py        新增 AsymmetricGasSensor
改动  dual_whisker_rl/envs/whisker_only_env.py    OU 湍流/meander、filament 密度、
                                                  非对称传感器接入、域随机化
改动  scripts/train_whisker_only_ppo.py           新增 --domain-randomization 开关
```

### 17.8 与外部参考的关系

当前 `DynamicPuffPlume` 本质上属于 Farrell 丝状/puff 羽流模型家族（filament 释放 + 随风平流 + 扩散 + 衰减 + 湍流蜿蜒）。后续若需进一步对标，可用开源 `pompy`（Farrell 模型 Python 实现）离线产场，验证本环境 whiff/blank 统计是否接近文献；若远期做 3D/移动机器人，可参考 `GADEN`（CFD 预计算风场 + filament 扩散 + 自带 MOX 传感器响应模型）。但当前 1m×1m 固定基座场景不需要 CFD，重点应放在间歇结构、观测对齐和域随机化。

## 18. 阶段三完成：whisker-only 环境观测硬件化

为了让仿真训练的触须策略能迁移到真机，本阶段把 `WhiskerOnlyPuffEnv` 的观测改成**只含硬件可获得特征**，并让仿真读数走**和硬件一致的预处理**。改造前的 12 维观测含硬件拿不到的特权信息——真实风向 `cos/sin(wind_rel)` 和到气源的真实距离 `source_distance`——若用它训练 PPO，策略会依赖这些量，一上真机观测分布对不上，基本白训。本阶段只改观测层，**不动力学、奖励、羽流模型、动作空间**（奖励 `get_reward(left,right)` 只用原始读数，天然不受影响）。

### 18.1 新增共享 observation builder

```text
dual_whisker_rl/observation/__init__.py
dual_whisker_rl/observation/whisker_observation_builder.py
```

`WhiskerObservationBuilder` 内部持有一个 `DualGasPreprocessor`（复用阶段一的硬件预处理模块，不改它），把左右传感器读数转成扣基线/平滑/趋势/差分特征，再拼接触须角度和扇区，输出 12 维观测。**仿真环境和将来的硬件 rollout 脚本都 import 它**，保证训练时和部署时观测语义一致（sim-to-real 观测对齐）。

12 维字段顺序（可行性文档 §9.3 简化版）：

```text
left_norm_signal, right_norm_signal,
left_norm_smooth, right_norm_smooth,
left_norm_trend,  right_norm_trend,
norm_smooth_diff, norm_trend_diff,
left_angle_norm,  right_angle_norm,      # 角度 / pi
left_sector_norm, right_sector_norm      # 扇区 / (sector_count-1)
```

刻意**不用** `DualSensorFeatures.as_policy_vector()`（它含噪声大的 `estimated_input`、且缺角度/扇区）。

### 18.2 环境改造要点

- `__init__` 新增参数：`observation_mode`（`"hardware"` 默认 / `"privileged"`）、`sim_sensor_scale`（默认 0.35）及一组 `preprocess_*` 参数；据此构造 `SensorPreprocessConfig` 和 `self.observation_builder`。观测空间在 hardware 模式用 builder 的有限边界，privileged 模式用 ±inf。
- **保留 privileged 模式做消融上界**：`observation_mode="privileged"` 时返回逐字保留的旧 12 维特权观测。可训一个带真实风向/气源距离的“上界”策略，论文里对照“硬件观测掉了多少性能”。
- **修复一个隐藏陷阱（关键）**：`scripts/visualize_whisker_only_env.py` 每帧会**额外**调 `env._observe()` 取 `whisker_state`。由于预处理器有状态（EMA/trend/baseline 累积），若把预处理更新放进 `_observe()` 会每帧双重更新、污染状态。因此把观测方法拆成三块：`_build_observation()`（有状态，reset/step 各调一次，推进预处理恰好一次并缓存 `_last_obs`）、`_current_info()`（无副作用组装 info）、`_observe()`（无副作用薄壳，只返回缓存观测，保持签名兼容可视化脚本）。
- `reset()` 调 `observation_builder.reset(init_baseline=None)`——用首帧读数锚基线，与硬件上电后在洁净空气锚基线一致，也对将来引入的每 episode 基线偏置随机化鲁棒。`step()` 记录**指令**扇区（硬件可得的“上一动作”）供观测使用。
- `scripts/train_whisker_only_ppo.py` 的 `run_metadata.json` 追加 `observation_mode`、`observation_field_names`、`sim_sensor_scale`；观测维度和历史堆叠维度本就动态计算，`ObservationHistoryWrapper` 无需改。

### 18.3 sim_sensor_scale 标定

仿真传感器读数量级 ~0-1.5，硬件 ADC ~500-4000，二者 `scale` 必须不同（builder/config 形状共享、scale 各自标）。仿真取 `scale=0.35`：验证 rollout 中 `norm_smooth` 洁净时 min≈0.05、in-plume mean≈2.6 / max≈3.35，落在 clip=5 以内不饱和，洁净接近 0——“无气味”和“有气味”清晰可分。**注意**：若改羽流量级或 DR 范围，应重新测 `norm_smooth` 分布并重标 `sim_sensor_scale`。硬件侧对应值为阶段一标定的 `scale=1500`。

### 18.4 端到端验证结论

已跑通 8 项验证：观测 12 维、字段名正确；连调 `_observe()` 两次预处理状态不变（**无双重更新**）、step 恰好推进一次；reset 帧传感器槽≈0；`norm_smooth` 不饱和且信噪可分；`visualize_whisker_only_env.py` 正常出图（真实触发额外 `_observe()`）；privileged 模式回归正常；训练 smoke（2000 steps）无形状错误，metadata 记录 `base=12 / stacked=120 / mode=hardware / scale=0.35`；域随机化下观测无 NaN（clip 生效，max|obs|=5）。

### 18.5 下一步

阶段三完成，触须 PPO 现在是硬件可部署观测形状。下一步可正式训练 whisker-only PPO（建议 `--history-length 20` ≈ 4s 历史、多 seed），并与固定/周期/随机策略在信息获取指标上对比（阶段四）。等硬件可用时，硬件 rollout 脚本直接复用 `WhiskerObservationBuilder`（换 `scale=1500`），保证观测与训练一致。机器人移动 + 触须联合 PPO 仍按路线图放到固定基座验证之后。

## 19. 初始位姿模式：让主动采样有可学信号

### 19.1 问题：奖励被“羽流存在”主导

首次正式训练（seed 1、200k、history 20、hardware 观测）后发现 `rollout/ep_rew_mean` 和 `eval/mean_reward` **全程持平**（分别约 54 和 51~57，无上升趋势），策略扇区分布收敛到固定偏好（左→3/4、右→5，熵 1.3~2.3，明显低于随机的 3.3，但未塌到 0）。结合诊断，判断根因是：初始位姿采样 `sample_robot_pose_in_plume()` 故意把机器人丢进浓度高分位（≥70%）的**密集区**，导致触须几乎每步都读到气味，`tracking_bonus`/presence 等常驻项贡献了大部分回报，**“触须指哪个扇区”对回报几乎没有影响**，主动采样缺乏可学梯度。也就是说，策略只学到了一个“还不错的固定采样姿态”，而非随气味变化的动态主动采样。

关键认识：决定“起始入羽流概率”的主要不是场地大小，而是这个初始位姿采样函数。

### 19.2 方案：加初始位姿模式旋钮（默认改为 plume_edge）

在 `WhiskerOnlyPuffEnv` 增加 `init_pose_mode` 配置（不改羽流场、不改奖励函数、不改动作空间，只改机器人落点分布，因此奖励量级和羽流间歇诊断不受影响）：

```text
in_plume    浓度密集区（旧默认，保留作对照）
plume_edge  羽流边缘/弱覆盖区（新默认）——取“活跃格子”（浓度>hit_threshold*0.25）
            浓度分布的 [init_edge_pct_low, init_edge_pct_high]=[10,50] 分位带，
            落在弱覆盖格子；活跃格子太少时退化为均匀采样
uniform     全场均匀（起始常在羽流外，信号最稀疏）
```

配套：`sample_initial_robot_pose()` 按模式分发；`sample_robot_pose_plume_edge()` 新增；`in_plume`/`edge` 共用 `_sample_from_grid_candidates()` 做带有效性检查的重试。训练 `run_metadata.json` 记录 `init_pose_mode`。

### 19.3 验证：起始入羽流率显著下降且信号变间歇

对每种模式 reset 60 次、看首步左右传感器 max 读数：

```text
in_plume  : 起始入羽流率 0.87   首步max mean 0.200  p10 0.068  p90 0.375
plume_edge: 起始入羽流率 0.52   首步max mean 0.095  p10 0.016  p90 0.200
uniform   : 起始入羽流率 0.40   首步max mean 0.091  p10 0.009  p90 0.227
```

`plume_edge` 把起始入羽流率从 87% 降到 52%、首步信号均值减半，且分布拉开（p10 近空白、p90 中等）——正是 whiff/blank 间歇、“触须指向影响信息获取”的场景，比 uniform 温和、不至于长期采不到气味。训练管线冒烟通过，默认已生效。

### 19.4 下一步

在 `plume_edge` 新默认下重新训练 whisker-only PPO（多 seed），并配合阶段四评估脚本用信息获取指标对比 ppo / random / periodic / fixed——只有在这个起始更间歇的场景里，才能真正检验主动采样相对固定/随机是否有增益。若仍无增益，再检查奖励是否需要进一步弱化常驻项、强化 trend/contrast。

## 20. 阶段四评估脚本 + 羽流稀疏化

### 20.1 触须采样策略评估脚本

新增 `scripts/evaluate_whisker_sampling_policies.py`：在同一批 plume seed 上公平对比 ppo / periodic / random_neighbor / 多个 fixed 角度，用**信息获取指标**（odor_hit_rate、mean_max_smooth、mean_abs_smooth_diff、mean_abs_trend_sum、reacquisition、sector_entropy 等）而非总回报评判。指标定义与动作序列生成**直接复用**硬件脚本 `compare_hardware_sampling_modes.py` 的 `compute_metrics`/`make_action_sequence`，保证仿真评估与硬件评估口径一致；指标从环境 hardware 观测槽读取。输出 `results/figures/whisker_policy_eval/`（metrics.json + summary.md + 柱状图）。

首轮评估（旧密集羽流、seed-1 模型）结论：命中率所有策略饱和到 0.92-0.96，PPO 未赢过最佳固定角度（fixed_5 的 max_smooth/contrast 反而更高）——即决策树"结论 C"。根因是奖励被"羽流存在"主导、采样位置对信息几乎无影响。

### 20.2 羽流稀疏化（让采样位置持续重要）

诊断发现 `plume_edge` 只改起始落点，但 300 步内动态羽流会扫过整场、固定基座照样被浸没（intermittency 0.73、whiff 2.5s/blank 0.9s）。因此把 `DynamicPuffPlume` 默认参数调稀疏+变窄：`puff_release_per_step 2→1`、`puff_init_sigma_crosswind 0.028→0.024`、`puff_init_sigma_downwind 0.060→0.052`、`decay_rate 0.045→0.052`、`max_puff_age 8→7`。并加 `plume_overrides` 覆盖机制（env cfg 可传，`DynamicPuffPlume(..., plume_overrides=...)`，在 `_dr_nominal` 之前应用，DR 围绕新值扰动），便于以后不改代码扫参。

before/after 诊断：intermittency `0.73→0.24`、whiff `2.5s→0.84s`、blank `0.92s→2.73s`（标准短 whiff/长 blank 结构）。重跑评估：命中率从 0.95 降到 0.71-0.87 且固定角度之间拉开，采样位置开始真正影响信息。但主动采样仍未明显赢过最佳固定角度（当前 PPO 是旧密集羽流训的、OOD，需在新羽流下重训才公平）。

## 21. 移动机器人触须气源搜索环境（MobileWhiskerPuffEnv）

把触须装到移动差速机器人上做气源搜索，复用 `WhiskerOnlyPuffEnv` 的全部新组件（稀疏羽流、AsymmetricGasSensor、硬件观测、舵机触须）。**注意**：路线图把移动机器人列为最后阶段（前提是固定基座已证明主动采样有增益，而该结论目前尚不明确）；本环境按用户要求提前搭建，严肃结论仍应回头以固定基座验证为前提。

### 21.1 环境设计

新增 `dual_whisker_rl/envs/mobile_whisker_env.py`，`MobileWhiskerPuffEnv(WhiskerOnlyPuffEnv)`：

- 动作 `MultiDiscrete([6, 10, 10])` = [移动, 左扇区, 右扇区]，与硬件命令兼容。
- 气源置于场内上风侧 `(-0.35, 0.0)`，机器人从下风侧随机起点、大致朝上风出发（`sample_initial_robot_pose` override），导航到源。差速底盘 `DifferentialDriveRobot(forward_speed=0.15, turn_rate=1.5, dt=0.2)`：0.03m/步、约 17°/步。
- **step 里先推进机器人位姿再采样触须**（触须点由 `whisker_model._point` 依位姿自动跟随）；父类 step 单体不可 hook，故整体重写（父类零改动）。
- 观测 = 父 12 维硬件特征 + 3 维本体感受（heading cos/sin、上一步移动动作），共 **15 维**；**不含**真实风向/气源方向等特权信息。奖励可用真实气源距离（仅训练期）。
- 奖励（初版）= 源搜索为主：`progress_reward_scale=15`（potential 式主项）+ `goal_bonus=10` + 小幅 trend/contrast。**问题**：progress 用真实距离、goal 必得，20k 冒烟就 0.9 成功率，策略可无视触须 → 见 §21.5 奖励重设计。

### 21.2 配套脚本与一处兼容修改

- `scripts/train_mobile_whisker_ppo.py`（复用 `ObservationHistoryWrapper`，history 20，PPO MlpPolicy）；`scripts/evaluate_mobile_whisker_ppo.py`（复用 `evaluate_policy`）；`scripts/visualize_mobile_whisker_env.py`（3 元动作、到达即停、goal 圆，复用 whisker 可视化助手）。
- `dual_whisker_rl/evaluation.py`：gymnasium 1.2.3 的 `gym.Wrapper` 不再转发属性访问，而评估工厂会套历史包装器，故把 `env.trajectory`/`env.goal_radius`/`env.hit_threshold` 三处改为 `env.unwrapped.*`（对旧 `PlumeEnv` 路径向后兼容）。
- 导出：`envs/__init__.py` 加 `MobileWhiskerPuffEnv`。

### 21.3 验证结论

- 环境：obs (15,)、action [6,10,10]；目标导向脚本控制器 20/20 到达、mean_return≈19（progress+goal 奖励符号正确）；随机策略 3/20（拉开）。历史堆叠 300 维。
- 可视化：羽流从场内源 `(-0.35,0)` 释放、随风飘向 +x、间歇结构在；机器人从下风侧导航到源、goal 圆正常。
- 训练冒烟（20k）跑通，metadata 记录 base=15/stacked=300/动作串/goal_radius/proprio 字段。
- 评估（20k 冒烟模型，10 回合）success_rate≈0.9。

### 21.4 关键风险与下一步

**风险**：progress 奖励用真实距离，导航梯度与羽流无关——本质是"带气味 bonus 的点导航"。20k 冒烟就 0.9 成功率，说明当前设置下策略可基本忽略触须、"主动采样"研究问题在此环境退化。要让触须真正起作用，需退火 progress 或做 odor-only 消融，并对比 hardware-obs vs privileged 上界。移动 sim-to-real 差距（理想底盘、假设可靠里程计）也需后续把速度/转率纳入域随机化。

**下一步**：正式训练 `--timesteps 300000 --n-envs 8 --history-length 20 --domain-randomization --seed 1` 并评估；同时补固定基座（whisker-only）的主动采样验证，作为移动结论的前提。

### 21.5 奖励重设计：让触须有可显现的效果

针对 §21.4 的风险（progress 主导使策略无视触须），重设奖励哲学：**把"触须端采到的气味"变成主导稠密项，特权 progress 降为弱引导**。新增 `odor_reach` 项（奖励 `max(left,right)`，依赖触须指向），并加重 trend/contrast。全部为可配置旋钮，新默认：

```text
progress_reward_scale 15 → 1.0    goal_bonus 10 → 4.0    oob_penalty 5
odor_reach_scale 0.7 (新增, clip 1.0)    mobile_trend_scale 0.3 → 1.5    mobile_contrast_scale 0.05 → 0.6
mobile_odor_hit_reward 0.02    mobile_time_penalty 0.01
```

odor_reach / trend / contrast 三项都依赖触须端读数 → 依赖触须指向。验证（同一"朝源导航"脚本控制器，只换触须子策略，30 seed）：

```text
固定触须(5,5)   mean_total_reward 15.35
周期扫描        mean_total_reward 16.65
随机扫描        mean_total_reward 17.08
```

动触须（扫描/随机）比固定触须多拿约 8-11% 回报，且差异稳定（std~2.8），说明触须指向现在对回报有可显现影响。random≥scan 因随机覆盖更多角度多抓间歇 whiff；训练好的主动策略应能靠智能瞄准超过两者。

**局限与定论路径**：这只证明"触须位置影响回报"。要定论"主动触须优于固定/扫描"，需在此奖励下训练 active-whisker PPO 与 fixed-scan-whisker 基线并对比（即课题主线 Joint > Fixed-whisker 结构）。若要触须更决定性，可进一步把 `progress_reward_scale` 设 0（气味驱动导航，触须成为找源必需），但训练更难。

### 21.6 扩大场地以显现追踪行为 + world_half 可配置

小场（1m）里机器人 ~20 步就到源，看不出追踪。把移动环境默认场地扩大到 **2m（±1.0）**，让机器人有一段需要沿羽流追踪的旅程。

- **`world_half` 做成可配置**：`DynamicPuffPlume` 和 `WhiskerOnlyPuffEnv` 都加 `world_half` 参数（默认 `WORLD_MAX`=0.5，即固定基座环境 **不受影响**，实测仍 ±0.5），类内 WORLD 常量用法改为 `self.world_min/world_max`。`MobileWhiskerPuffEnv` 默认 `world_half=1.0`。可视化脚本改从 `env.get_map_metadata()` 读世界边界，不再硬编码。
- **关键配套（大场需让羽流跨场）**：单纯放大场地会让下风起点闻不到气味。因此移动环境同时把 `wind_speed_range` 提到 `(0.12,0.20)`、`plume_overrides={"max_puff_age":16}`（puff 寿命延长，飘得更远），源置 `(-0.8,0)`（上风侧场内），起点在下风半场 `x∈(0.35·wh, wh-0.15)`，`max_steps 200→400`。
- **验证**：沿 x 轴（y=0）平均浓度/命中率 —— 源区 x=-0.8: 1.10/0.99，下风起点区 x=0.9: 0.077/0.41，全程平滑梯度、间歇性沿程增强，是可追踪羽流。目标导向脚本控制器 20/20 到源、平均 48 步（约 1.5m 旅程，小场仅 ~20 步）。可视化确认 2m 场内 filament 成缕横穿。

注意：真正的"追踪现象"仍需**训练策略**才能显现（随机策略会乱转出界）；本节只是把环境/羽流准备到能显现追踪的 regime。

## 22. 时序编码器可切换：Transformer / GRU / MLP（mobile）

`ObservationHistoryWrapper` 把 20 帧观测堆叠成扁平向量（mobile 15×20=300）后，用哪种网络在**策略内部**恢复时间维并编码，做成 `train_mobile_whisker_ppo.py --temporal-encoder` 三选一：

- **transformer**（`dual_whisker_rl/agents/transformer_history.py`，先前 commit 加入）：CLS token + 可学习位置编码 + 2 层 4 头 encoder，取 CLS 输出。配套注意力可视化脚本 `analyze_mobile_transformer_attention.py`。
- **gru**（本节新增，`dual_whisker_rl/agents/gru_history.py`）：单层 GRU（默认 hidden=64），取最后一层最后时刻隐状态 `h_n[-1]` → Linear+GELU → features_dim。
- **mlp**：SB3 默认 flatten 提取器，扁平 300 维直接进 `net_arch=[160,160]`。

### 22.1 为何加 GRU

Transformer 在本任务"低维（每帧 15）+ on-policy 小样本 + 慢响应传感器"场景下归纳偏置弱、额外参数（CLS/位置编码）多、训练不稳。MQ-3 是慢响应、强自相关信号，天然序列递推结构——单层 GRU 顺序内建、无需位置编码、参数少（实测提取器 ~1.97 万参数），PPO 下更稳，作为 transformer 的平级替代/对照。

### 22.2 实现要点

- `GRUHistoryExtractor(BaseFeaturesExtractor)` 与 transformer 提取器接口一致、可直接互换（同样校验扁平维 == history×base、reshape `[B,H,base]`）；不实现注意力接口。
- 训练脚本：`--temporal-encoder` 增 `gru`；新增 `--gru-hidden-size/layers/dropout/features-dim`；`build_policy_kwargs` 加 gru 分支；resume 类型检查从"是否 transformer"泛化为按 encoder 映射到对应提取器类（mlp=默认 flatten）；`run_metadata` 新增 `gru` 字段。输出路径已按 `mobile_whisker_{encoder}_ppo` 自动隔离（gru 独立模型/日志目录）。
- **未覆盖**：whisker-only 训练脚本；GRU 无注意力，未做替代可解释性工具。

### 22.3 验证（冒烟，非结论）

三种 encoder 均端到端跑通、各自写独立模型：gru 3000 步（fps~617）、transformer/mlp 各 1200 步均正常收尾；gru 的 `run_metadata.json` 正确记录 `gru={hidden_size:64,n_layers:1,dropout:0.0,features_dim:64}`、`features_extractor_class=GRUHistoryExtractor`。

**待做（真正判断效果）**：相同 seed/timesteps 下 gru / transformer / mlp 三方 A/B，比 `ep_rew_mean` 收敛与稳定性 + evaluate 指标。**特别注意**：观测每帧已含 `norm_trend/norm_smooth/diff` 等手工时序特征，时序编码器未必是瓶颈——务必带上 mlp 基线，避免"换架构其实无差别"。

## 23. Mobile 奖励稳定性修正 + TensorBoard 奖励拆分

### 23.1 成功率后期下降的奖励侧诊断

已有长训练显示两类现象：GRU 训练窗口的 `rollout/success_rate` 可升到约 0.61，但仅 10 episode 的 eval 成功率在 0.0～0.5 间剧烈波动；Transformer 后期则真实塌缩到连续 0 成功。除评估样本量过小外，旧奖励还存在两个确定的局部激励问题：

- `progress_reward_scale=1.0` 时，朝气源正常前进约 0.03m 只得到约 `+0.03`，低于每步 `time_penalty=-0.05`，即正确的非终止移动仍可能净亏分，学习主要依赖稀疏的 `goal_bonus=50`。
- `oob_penalty=5`，而 400 步超时累计时间成本为 `20`；尽快越界终止可能比留在场内超时更划算，形成提前结束 episode 的漏洞。

### 23.2 新默认与安全约束

`MobileWhiskerPuffEnv` 默认奖励调整如下：

```text
progress_reward_scale              1.0  -> 3.0
best_concentration_reward_scale    5.0  -> 3.0
mobile_time_penalty                0.05 -> 0.03
stagnation_window                  5    -> 20
stagnation_penalty                 0.10 -> 0.02
oob_penalty                        5.0  -> 25.0
goal_bonus                         50（不变）
```

这样，朝气源前进 0.03m 的无气味单步约为 `3×0.03-0.03=+0.06`，提供连续的正确方向反馈；浓度历史最佳增量仍保留，但降低权重，防止气味辅助项盖过导航目标。停滞惩罚延后并减弱，避免正常转向或短暂等待间歇羽流时被过度处罚。

新增两项初始化安全检查：正辅助奖励理论上界不得超过 `goal_bonus/4`；`oob_penalty` 必须不小于“最大 episode 时间成本 + 正辅助奖励上界”。当前分别为 `11.4853`、`12.0`，故越界惩罚至少需 `23.4853`，默认取 25。显式传旧值 5 会直接报错，避免静默恢复奖励漏洞。

### 23.3 TensorBoard 奖励拆分

训练回调从每个 step 的 `info["reward_components"]` 累计并写入 TensorBoard。六个实际奖励分量均有两种曲线：

- `reward_components/<name>_step_mean`：当前 rollout 内逐步均值，观察即时强度与正负号。
- `reward_components/<name>_episode_sum_mean`：当前 rollout 内完整 episode 的累计均值，观察该项对整局回报的贡献。

奖励项为 `distance_potential_progress`、`best_concentration_improvement`、`time_penalty`、`stagnation_penalty`、`goal_bonus`、`out_of_bounds_penalty`。另加 `reward_components/total_step_mean` 和 `reward_components/total_episode_sum_mean`，用于核对拆分之和；run metadata 同步记录最大时间成本和最小安全越界惩罚。

### 23.4 验证（连通性，不是训练结论）

- 200 step 随机环境检查：每一步环境返回 reward 与六项之和在 `1e-12` 误差内一致。
- 1024 step、2 env、GRU smoke 跑通；TensorBoard 实际包含 14 条奖励曲线（6×2 + total×2），成功 episode 的拆分总和示例为 51.98。
- 旧 `oob_penalty=5` 被安全校验拒绝；Python compileall 通过。

下一轮正式训练应使用新 run（奖励语义已变化，不建议从旧 checkpoint 续训），把 eval episode 数从 10 提高到至少 50，并同时观察 `eval/success_rate`、`rollout/success_rate`、`goal_bonus_episode_sum_mean`、`out_of_bounds_penalty_episode_sum_mean` 与 `time_penalty_episode_sum_mean`。脚本跑通不能证明成功率不会再次下降，需要多 seed 曲线确认。

## 24. 本地 / 云端统一的项目相对路径

为避免 Windows 盘符、用户目录或 Linux 云主机工作目录进入 CLI 默认值和运行元数据，新增 `dual_whisker_rl/paths.py`，统一三种语义：

- `project_path(path)`：相对路径始终按仓库根目录解析，不依赖启动时的 current working directory；显式绝对路径保持不变，支持云盘挂载点。
- `resolve_path_args(args, ...)`：集中解析 argparse 中的输入、模型、日志、图像、checkpoint 等路径字段。
- `portable_path(path)`：仓库内路径在 JSON metadata 中写成 POSIX 风格相对路径，例如 `results/models/mobile_whisker_gru_ppo.zip`。

所有 Python 入口中的 `default=ROOT / ...` 和 `ROOT / ...` 输出拼接已改为相对默认值，并覆盖训练、评估、环境可视化、注意力分析、羽流诊断、硬件采集/分析及基础 rollout 脚本。`Path(__file__).resolve()` 只保留作运行时定位仓库和注入 Python import path，不包含机器特定的硬编码目录。

验证：

- 全仓扫描没有发现 `C:\\...`、`D:\\...`、`/home/...`、`/Users/...` 等硬编码项目路径（第三方许可证中的 URL 除外）。
- `compileall` 通过；从仓库外的系统临时目录调用 17 个脚本的 `--help` 全部成功。
- 从系统临时目录实际启动 256-step mobile GRU smoke，模型仍落到仓库 `results/smoke/cloud_path_model.zip`。
- smoke 的 `run_metadata.json` 中 model/TensorBoard/checkpoint 路径均为 `results/...`，无盘符、无用户目录。

README 已增加 Linux 云服务器建环境、后台训练、TensorBoard SSH 隧道和训练产物回传示例。`results/` 仍保持 gitignore；释放云实例前必须单独同步模型与日志。

## 25. Mobile 主动触须诊断 + blank_age + 可归因重捕获奖励

针对 mobile 策略在无气味时倾向旋转底盘、没有优先利用可动触须扫描的问题，先加入可观测、可量化且有上限的主动触须信号，不在本阶段强制限制底盘动作。目标是区分“触须扫描找到气味”和“底盘旋转或羽流自然扫过”，避免只看总成功率时误判触须贡献。

### 25.1 blank_age 观测与旧模型兼容

`MobileWhiskerPuffEnv` 默认在原 15 维硬件可部署观测末尾追加 `blank_age_norm`，单帧变为 16 维。`blank_age` 从未命中气味的第一步开始累计，任一传感器达到 `hit_threshold` 时归零；归一化形式为 `min(blank_age_s / 10.0, 1.0)`。该量仅依赖传感器读数和本地计时，真机可重建，不是仿真特权状态。

配置 `include_blank_age_observation=false` 可恢复旧 15 维接口，用于复评已有 checkpoint。默认 16 维下，history=20 的堆叠维度由 300 变为 320，不能直接续训旧模型；奖励语义也已变化，正式训练应新开 run。

### 25.2 严格重捕获与奖励归因

有效重捕获要求：本 episode 先前已有 odor hit，随后连续 blank 至少 `reacquisition_min_blank_s=1.0`，再重新达到 hit 阈值。初次发现气味不算重捕获，短于 1 秒的阈值抖动也不计入严格事件。

重捕获前 `reacquisition_credit_window_s=1.0` 内的动作分为三类，且互斥：

- `whisker_only`：触须动过，底盘所有动作均为 `stop`；
- `body_assisted`：底盘出现任意非 `stop` 动作，包含前进、转弯和原地旋转；
- `passive`：底盘与触须均未运动。

只有 `whisker_only` 获得 `whisker_reacquisition_bonus=0.25`。单局最多奖励 `max_whisker_reacquisition_rewards=4` 次，故理论总上限为 1.0，防止策略主动反复丢失/重捕获刷分。严格重捕获事件会清零 stagnation，但底盘辅助或被动重捕获不获得触须专属 bonus。

新奖励上限计入原安全约束：默认正辅助奖励理论上界由 11.4853 增至 12.4853，仍小于 `goal_bonus/4=12.5`；最大时间成本 12.0，因此最小安全越界惩罚为 24.4853，默认 `oob_penalty=25` 仍满足要求。

### 25.3 诊断输出

环境在每步 `info` 和 trajectory 中记录 odor hit、blank 时长、近期底盘/触须运动、严格重捕获类别和实际奖励事件。训练 TensorBoard 的 `diagnostics/` 与离线评估 JSON 同时输出：blank 期间底盘旋转/任意运动/触须运动比例、blank_age 均值和最大值、严格重捕获数量及每千步频率、三种归因的数量和占比、实际获得触须奖励的次数。旧 `reacquisition_events` 和 `mean_reacquisition_time` 保留原口径，便于历史结果比较。

### 25.4 连通性验证（不是策略效果结论）

- 确定性状态序列覆盖首次命中、短 blank、触须独立重捕获、底盘辅助、被动重捕获和奖励封顶；三种归因严格互斥，仅触须独立事件获奖，6 次合格尝试中只有前 4 次获奖，总计 1.0。
- 默认/兼容观测分别为 16/15 维且均满足 Gymnasium Box；随机环境 300 step 中每步 reward 与 7 个分量之和误差小于 `1e-12`。
- 256 step、2 env、GRU smoke 训练完成；base/stacked 维度为 16/32，TensorBoard 包含奖励分量与 `diagnostics/` 标签，固定种子评估 JSON 正常输出全部新指标。
- 旧 `mobile_whisker_gru_ppo_9300000_steps.zip` 的堆叠维度为 300；默认 16 维环境明确不兼容，关闭 blank_age 后可按 15×20 正常恢复并产生动作。

这些检查只证明状态机、奖励核算和日志链路正确。下一步应从新模型开始做同 seed 消融：`whisker_reacquisition_bonus=0.25` 对比 0，同时比较成功率、blank 底盘旋转比例、触须独立重捕获率和样本效率，不能用总回报单独判断奖励有效性。

## 26. 实验动画默认迁移至 H.264 MP4

为降低动态羽流、轨迹和注意力动画的存储占用，所有活动动画入口默认由 Pillow GIF 改为 FFmpeg/libx264 MP4。共享保存函数根据扩展名选择编码器：`.mp4` 使用 CRF 21、medium preset、`yuv420p` 和 faststart；`.gif` 仍使用 PillowWriter。编码滤镜会把奇数宽高补到偶数，解决 whisker-only 动画 1365×1105 无法直接编码为 yuv420p 的问题。FFmpeg 是系统依赖，不加入 Python requirements；不可用时明确提示安装并加入 PATH，不静默退回 GIF。

移动评估、移动环境和 whisker-only 环境的正式参数改为 `--animation-path`，旧 `--gif-path` 作为兼容别名保留。扩展名决定格式，因此旧工作流仍可显式请求 GIF。固定触须/联合 PPO 评估和 Transformer 注意力动画也已改为默认 MP4；已有 GIF 不做转换或删除。

验证结果：新增的 4 个 unittest 全部通过；移动、whisker-only 和注意力三种渲染器均实际输出 H.264/yuv420p，其中奇数尺寸被正确补成 1366×1106。对同一段 20 帧移动羽流轨迹，MP4 为 242,536 bytes，GIF 为 3,836,819 bytes，GIF 约为 MP4 的 15.82 倍。六个相关脚本的 `--help` 均正常。现有 Transformer 模型仍是旧的 300 维堆叠观测，与当前默认 320 维环境不兼容；注意力动画编码使用渲染函数独立验证，这一旧模型兼容问题与本次格式迁移无关。

## 27. E4 统一仿真对比运行器

本节实现了 E4 仿真主对比实验的统一运行链路：`dual_whisker_rl/e4_experiments.py` 提供固定场景库、方法动作适配、B0/B1 规则控制器、B6 单侧屏蔽和逐场景评估；`scripts/run_e4_experiments.py` 提供 `prepare/calibrate/train/evaluate/summarize/run/status` 阶段。

正式场景按 3 风速 × 3 距离 × 2 横偏 × 3 源强 × 4 初始朝向 × 3 重复生成测试集 648 个场景，验证集 216 个场景；smoke 每个 split 取 12 个场景。场景通过 `reset(options={"scenario": ...})` 注入，源位置、位姿、风参数和源强记录在 JSONL 中。每个方法×种子逐场景原子保存，摘要同时给出分层指标和与主方法的配对差值。

M/B2/B3/B6 使用 GRU 历史策略，B4 使用单帧 MLP，B5 通过输入维度启发式选择等宽 MLP 并记录实际宽度；B0/B1 不训练。B6 的缺失侧特征、几何、左右差分、命中、重捕获和气味奖励均屏蔽。 B1 阈值与搜索/扫描持续时间由配置控制，formal calibrate 在验证集搜索并封存 `calibration_b1.json`。B2 formal 校准会为十个固定扇区分别训练两个种子，在验证集按成功率、最终距离和 checkpoint 顺序选择；smoke 明确记录为未校准。

恢复以状态文件、checkpoint、代码摘要、配置摘要和场景摘要为边界。 场景 JSONL 会再次和 manifest 做 hash 校验；评估每个 split 另写状态文件，支持逐场景 resume。恢复模型保留 PPO 参数与优化器，但环境重新初始化，metadata 明确记录续训来源；缺失场景不被计为失败。smoke 结果只用于连通性验证，不作为方法优劣结论。

## 28. 自研双触须主控 PCB 原理图接口冻结

为避免原理图绘制人员继续从开发板飞线记录中猜测板型、电源或连接器针序，新增 `docs/PCB_SCHEMATIC_INTERFACE_SPEC.md`，作为自研 PCB 的唯一权威输入。原有硬件模块图继续用于论文和系统概念说明，`hardware/wiring/wiring_notes.md` 与固件 README 继续保留开发板 smoke test 接线方式，但都显式指向新规范。

### 28.1 已冻结的板级方案

- 主控为 STM32F103C8T6/LQFP48，沿用当前 HSI 8MHz 和固件外设分配：PA0/PA1 为 TIM2 双舵机 PWM，PA2/PA3 为双路 MQ-3 ADC，PA9/PA10 为 USART1，PA13/PA14 为 SWD。
- PC 接口改为板载 USB-C + CH340C，CH340C 按 3.3V 供电并使用内部时钟。串口协议保持 `STEP left right`、115200/8N1 和一问一答 JSON，不改上位机软件接口。
- 外部稳压 5V（建议持续能力不低于 3A）供舵机和 MQ-3。USB VBUS 与外部 5V 经肖特基二极管或入后只生成逻辑 3.3V；USB 单独接入时 MCU/CH340C 可工作，但舵机和 MQ-3 保持断电。
- J3/J4 舵机接口固定为 `GND / +5V_SERVO / PWM`；J5/J6 MQ-3 接口固定为 `+5V_SENSOR / GND / AO / DO`；J7 SWD 固定为 `3V3 / SWDIO / SWCLK / NRST / GND`。左右接口不可互换。
- 两路 MQ-3 AO 使用 6.8kΩ/10kΩ、1% 分压和 100nF 滤波后进入 PA2/PA3；在模块 AO=5.25V 的最坏供电假设下，ADC 节点约 3.125V，计入电阻容差仍低于约 3.15V。DO 使用相同分压预留到 PB0/PB1，当前固件不启用。

### 28.2 首版取舍

首版定位为“最小验证板”，包含 MCU 最小系统、USB 串口、电源或入、ADC 分压、去耦、复位/启动、SWD 和测试点，不加入输入保险丝、防反接 MOS、TVS、负载开关或电流检测。USB D+/D- 的 ESD 器件只预留焊盘并标记 DNP。该取舍减少首板复杂度，但要求使用带限流的实验电源、严格检查 J2 极性，不能将此版本直接视为量产级硬件。

### 28.3 原理图交付与验收原则

新规范给出 J1～J7 的固定针序、网络命名、建议 BOM、ERC 审查清单和首板上电顺序。原理图验收必须覆盖 USB 单供电、外部 5V 单供电和双电源同时接入三种状态，并核对 USB 不向外设支路供电、两路 ADC 不绕过分压、CH340C 收发方向不交叉。制板后按“电源空载 → USB 枚举 → SWD → 串口 → ADC 已知电压 → 单舵机 → 双舵机双传感器”顺序逐级验证。

## 29. E4 场地边界配置化与大场羽流适配

本轮将仿真场地边界统一为配置项 `world_bounds: [min, max]`，当前 E4 配置为 `[-2.5, 2.5]`。`world_half` 作为旧入口兼容别名保留；显式边界要求关于原点对称，避免场景缩放、奖励归一化和羽流裁剪出现不一致。

`DynamicPuffPlume`、`PaperPuffPlume`、`WhiskerOnlyPuffEnv` 和 `MobileWhiskerPuffEnv` 共享解析后的 `world_min/world_max`。E4 场景生成按场地半宽缩放源距离、下风距离和横风偏移，并在生成时检查源点与机器人均位于有效边界内；评估越界指标不再写死为 `abs(x)>=1.0`。动态移动环境的 puff 默认寿命按场地直径/最小风速估算，±2.5 场地下约为 45.8 s，保证羽流能够传播到下风侧区域。

验证：E4 formal 场景数量仍为 648（validation 216），场景坐标和 plume 网格均落在 ±2.5；E4、paper 和移动场景初始化相关 17 项测试通过。低层移动环境未显式设置边界时继续使用旧 ±1.0 默认，正式 E4/配置文件使用新范围。

## 30. E4 正式训练改用多进程环境并行

`run_e4_experiments.py` 新增 `vec_env_backend`/`--vec-env-backend`，支持 `dummy`、`subproc` 和 `auto`。smoke 默认保留 `DummyVecEnv`，便于直接调试；formal 默认改为 `SubprocVecEnv`，8 个训练环境分别运行在独立 Python 进程中。Windows 显式使用 `spawn` 并在主入口调用 `freeze_support()`，避免子进程递归执行命令入口。

训练代码不再依赖只有 `DummyVecEnv` 才有的 `env.envs[0]`。堆叠观测维度改从 VecEnv 的公开 `observation_space` 读取，GRU 的单帧基础维度通过 `get_attr("base_observation_dim")` 从各 worker 获取并检查一致性。并行后端写入 manifest、恢复签名和每个模型的 metadata，因此使用不同后端的旧任务不会被 `--resume` 静默混用。固定场景评估仍在主进程逐场景执行，保持原子保存和配对复现逻辑不变。

验证：`py_compile` 和 E4 核心 4 项单元测试通过；Windows `spawn` 下以 2 个 worker 分别完成主方法 GRU 和 B5 历史 MLP 的 64-step smoke。主方法产物记录 `actual_timesteps=64`、`vec_env_backend=subproc`、`subproc_start_method=spawn`，模型与 TensorBoard event 均生成；同签名 `--resume` 正确跳过完成任务。`auto` 在 `n_envs=1` 时回退到 `dummy`，formal 默认仍解析为 8 worker 的 `subproc`。这些检查验证运行链路，不代表训练效果。

## 31. E4 训练剩余时间估算

正式训练增加 rollout 周期计时回调。计时从一次环境 rollout 开始，到该批数据完成配置的 5 个 PPO epoch 更新并准备开始下一次 rollout 为止；最后一轮在训练结束回调中结算。每轮终端输出本轮耗时、最近 5 轮平均耗时、实际完成步数和预计剩余分钟数。剩余轮数按 `ceil((目标总步数-模型实际步数)/每轮步数)` 计算，因此与 SB3 按完整 rollout 向上对齐的预算规则一致。恢复训练继续使用 checkpoint 中的 `model.num_timesteps`，不会从零估算。

周期耗时使用单调墙钟计时，因而会自然计入该轮发生的 checkpoint 或周期验证开销，更接近用户实际等待时间。平滑窗口、单位、PPO epoch 数同步写入每个训练任务的 `metadata.json`。ETA 以及准备、训练、评估和汇总等面向人的终端状态提示统一使用中文；结构化 JSON 字段保持稳定，避免破坏自动化读取。重定向到日志或任务捕获通道时显式使用 UTF-8，交互式 Windows 终端则沿用本地编码，避免中文乱码。验证采用 B4、2 个 SubprocVecEnv worker、2048 总步数完成两个 rollout：第一轮耗时 1.36 秒、剩余估算 0.02 分钟，第二轮完成后为 0.00 分钟；E4 核心 4 项单元测试和语法检查通过。这只是计时链路验证，正式服务器上的数值会随 CPU、worker 数和周期验证而变化。

## 32. 服务器训练前的路径可移植性审计

对主仓库及 `active-olfaction-pape` 论文子仓库的 Python、YAML、JSON、TOML、Shell、PowerShell 等运行文件进行了绝对路径复扫。训练主线和 E4 已统一使用 `dual_whisker_rl.paths`，没有本机盘符、用户名、家目录或挂载目录字面量。残留项来自辅助工具：两份网络结构图脚本写死 Windows 中文字体、Puppeteer 配置写死 Edge、PDF 文本提取工具写死桌面文献目录，论文子仓库的两份导出脚本也写死 Windows 字体。本轮已全部移除。

新增 `dual_whisker_rl/font_utils.py`，图表脚本优先读取 `DUAL_WHISKER_FONT_REGULAR`/`DUAL_WHISKER_FONT_BOLD`，否则按字体族自动发现 Microsoft YaHei、Noto Sans CJK、Source Han Sans 等中文字体；论文子仓库提供等价工具和独立环境变量。PDF 提取脚本改为接收 `名称=PDF路径` 参数，Puppeteer 交由运行环境发现浏览器。字体和文献可以在运行时使用绝对路径，但仓库代码不再保存任何机器位置。

新增 `scripts/check_portable_paths.py` 作为上传前检查，扫描 Windows 盘符、Linux/macOS 用户与挂载目录、UNC 网络路径。`.venv`、`results`、`node_modules`、STM32 `build` 和本地 IDE 配置属于 Git 忽略的环境或生成产物，不纳入源码审计。实际运行审计通过；从系统临时目录启动 `train_fixed_whisker_dqn.py`、`train_joint_ppo.py`、`train_mobile_whisker_ppo.py`、`train_whisker_only_ppo.py` 的帮助入口均成功，E4 formal dry-run 仍正确解析 216/648 场景、5 个种子、8 个 `subproc` worker。进一步从系统临时目录实际完成 B4、2 worker、64-step 训练，生成的 `metadata.json` 和 `state.json` 均不含 Windows 盘符，模型和 TensorBoard 路径保持为 `results/...`。`train_joint_dqn.py` 是已废弃入口，会按设计立即提示使用 PPO。修改文件均通过 Python 编译检查，Windows 字体自动发现得到有效字体文件。

## 33. 服务器训练文件与依赖分层

从所有 `train_*`、`evaluate_*` 和 E4 统一入口的导入链复核依赖。E4 正式训练与批量评测的直接第三方依赖为 NumPy、Gymnasium、PyYAML、PyTorch、Stable-Baselines3 和 TensorBoard；旧评测绘图直接使用 Matplotlib。PySerial 只服务真机串口，FFmpeg 只服务 MP4 动画，因此二者均不属于 E4 批量实验运行条件。新增 `requirements-server.txt` 汇总仿真与强化学习依赖，服务器可通过一个 requirements 入口安装。

新增 `docs/SERVER_TRAINING_MANIFEST.md`，固定 E4 最小上传集、推荐验证文件、可排除目录和恢复流程。正式运行只需要 `dual_whisker_rl/`、E4 入口、smoke/formal 配置与 Python 依赖文件；`.venv`、首次运行时的 `results`、硬件/STM32、论文仓库、文档输出、临时 PDF、画图动画和本地 IDE 状态都可从训练上传包中排除。续训所用 E4 输出目录（默认 `results/e4/`）属于恢复输入，不能按首次上传规则丢弃；smoke/formal 长期并存时应显式使用不同 `--output-dir`。

主仓库 `.gitignore` 现显式覆盖 Python 环境与缓存、`results`、STM32/CMake `build`、Git 元数据、本地 IDE/助手目录、日志和本地上传归档；论文子仓库同步忽略本地环境与编辑器状态。`.git` 元数据本身不会被 Git 跟踪，该规则主要用于工作区复制和归档清单的统一表达。

## 34. 论文仓库与训练主仓库解耦

`active-olfaction-pape` 已从 `dual-whisker-rl` 的 Git 索引中移除，主仓库不再保存论文仓库的 gitlink，也不负责传播论文仓库提交。主仓库 `.gitignore` 显式忽略同名目录，允许本地继续保留独立论文工作区而不影响训练代码状态。论文目录的文件和本地提交均未删除；后续如需同步论文，应在论文仓库中独立完成。

## 35. Conda 服务器一键部署

新增 `scripts/setup_server_conda.sh`，面向已安装 Conda 的 Linux 训练服务器。默认创建或复用 `dual-whisker-rl` 环境和 Python 3.11，通过 `requirements-server.txt` 安装仿真、PyTorch、Stable-Baselines3 与 TensorBoard；部署后验证核心包版本、CUDA 状态、E4 单元测试、路径可移植性和 formal dry-run，并把实际包版本保存到 `results/deployment/`。

脚本未检测到 NVIDIA GPU 时安装 CPU 版 PyTorch；检测到 GPU 时使用项目常规依赖安装，不根据驱动版本猜测 CUDA wheel。服务器需要固定 wheel 时显式传 `--torch-index-url`，需要强制 GPU 可用时传 `--require-cuda`。真机串口与 FFmpeg 继续作为 `--with-hardware`、`--with-ffmpeg` 可选项，正式批量 E4 默认不安装。`.gitattributes` 固定 shell 脚本使用 LF，避免 Windows 工作区提交后在 Linux 出现解释器换行错误。

## 36. E4 正式实验一键启动与完整恢复

新增 `scripts/run_e4_formal_conda.sh`。脚本定位项目根目录、加载 Conda 初始化脚本并激活 `dual-whisker-rl`，固定 formal profile 与配置，默认执行完整 `run`，并始终传入 `--resume`。支持 `prepare/calibrate/train/evaluate/summarize/status/stop` 分阶段调用、后台 nohup、统一日志、PID 记录和 `flock` 单目录互斥。`stop` 核对 PID 对应 E4 进程后发送 SIGINT，使运行器走中断状态保存路径。首次运行和中断恢复使用同一命令；新实验通过新的 `--output-dir` 隔离，不提供覆盖旧结果的快捷开关。

补齐正式校准恢复边界：B1/B2 完成产物新增由场景摘要、配置、前两个校准种子、并行参数和关键代码摘要组成的签名。匹配时跳过已完成的 B1 规则搜索和 B2 角度搜索；产物损坏或签名不一致时拒绝复用。B2 未完成时继续沿用候选角度各自的 checkpoint 与逐场景恢复。另修正 E4 CLI 默认 profile 的 `foramal` 拼写错误，默认恢复为 smoke，并增加运行器测试覆盖默认值、校准签名匹配及损坏文件拒绝。

校准过程新增中文进度与 ETA。B1 在 48 个规则候选开始和完成时打印参数、成功率、最终距离、累计耗时与预计剩余分钟；B2 在 10 个固定扇区 × 2 个种子的训练/验证开始和完成时打印总体任务进度，各 PPO 短训练内部继续输出 rollout ETA。逐场景评测每 50 个场景（不足 50 时在末尾）打印完成数、耗时和剩余分钟。`run --resume` 读取到签名一致且状态为 completed 的 `calibration_b1.json`、`calibration_b2.json` 时分别打印跳过信息，并在两者都完成后直接进入正式训练。

正式服务器首次运行暴露出 `--max-steps` 未在命令行显式传入时仍保留为 `None`，而校准签名直接执行 `int(args.max_steps)`，导致在任何校准或训练开始前中断。参数解析现在会优先采用命令行值，否则读取 YAML 的 `max_steps`，配置也未提供时使用兼容默认值 400，并统一校验为正数；校准签名仍保留配置回退，防止未经过完整 CLI 解析的内部调用再次触发同类错误。该修复不会改变已生成的场景，也不会损坏 checkpoint；相同输出目录可直接用 `--resume` 重试。

随后发现校准签名包含整份 `run_e4_experiments.py` 的文件哈希，导致上述参数修复本身也会让已完成的 B1/B2 校准被判为不兼容。签名升级为 `e4-calibration-v2`：覆盖配置、场景摘要、校准搜索空间、种子、预算和核心环境代码，但不再绑定进度打印、命令行胶水等整份运行器文本。已知提交 `14fe05c` 与当前校准语义一致，其旧签名可被精确识别并原子迁移；未知旧签名仍明确拒绝，防止跨配置复用。

服务器上的 B1 文件实际由未形成 Git 提交的中间运行器生成，旧脚本哈希无法从提交历史还原。对此增加第二层严格迁移：只接受未带签名版本号的 `e4-v1` 完成产物，并逐项校验 48 个 B1 候选或 10×2 个 B2 任务的候选集合、指标有限性、种子、102,400 步预算及按当前规则重算的最优项。完整内容全部一致时才原子改写为 v2 签名，并记录 `migration_validation`；缺项、损坏、选择结果不一致以及已经是 v2 但签名不符的产物继续拒绝。

同类问题还影响 `prepare` 写入的训练 `state.json`：旧签名包含整个运行器文件，脚本修复后会把尚未开始、`actual_timesteps=0` 的 prepared 状态误判为不可恢复。训练签名升级为 `e4-training-v2`，只覆盖环境/策略核心代码、配置、并行方式和 PPO 语义超参数，不再绑定进度打印或 checkpoint 保存频率。恢复时若任务没有训练步数、checkpoint、最终模型或 metadata，则原子刷新准备状态签名并开始训练；只要已有任一训练产物，场景或签名不一致仍严格拒绝。

## 37. 独立 PPO 训练入口改用多进程环境

`train_joint_ppo.py`、`train_whisker_only_ppo.py` 和 `train_mobile_whisker_ppo.py`
此前虽然支持多个环境，但实际使用 `DummyVecEnv`，所有场景仍在主 Python 进程内顺序推进。
三个入口现统一增加 `--vec-env-backend auto|dummy|subproc`：默认 `auto` 在
`n_envs>1` 时解析为 `SubprocVecEnv`，每个训练场景进入独立进程；单环境回退为
`DummyVecEnv`。子进程跨平台统一使用 `spawn`，Windows 主入口调用
`freeze_support()`。

周期评估环境使用与训练环境一致的 VecEnv 类型，避免 SB3 训练/评估后端类型不一致；
mobile 脚本不再访问仅 `DummyVecEnv` 暴露的 `env.envs[0]`，而是从公开
`observation_space` 和 history length 推导单帧维度。实际后端与子进程启动方式写入
`run_metadata.json`，终端也会打印实际后端。`dummy` 仍保留用于断点调试和低开销 smoke。

验证：三个入口均通过 Python 语法和帮助入口检查；Windows `spawn` 下各以 2 个 worker
完成一次训练，mobile 为 64 步，joint/whisker-only 因固定 512-step rollout 各为 1024
全局步。三个终端输出和 metadata 均记录 `vec_env_backend=subproc`、
`subproc_start_method=spawn`。这些检查验证并行链路，不代表策略效果。

## 38. Mobile 独立训练默认 800 万步与轮次 ETA

`train_mobile_whisker_ppo.py` 的默认训练预算由 300,000 步提高到 8,000,000 步。
`--timesteps` 仍可显式覆盖；从 checkpoint 恢复时保持原语义，表示额外训练步数。

新增完整 PPO 训练轮次计时：一轮从环境 rollout 开始，到该批样本完成全部 PPO epoch
更新并准备进入下一轮为止。每轮开始打印轮次、墙钟时间和该轮环境步数；每轮结束打印
实际耗时、最近 5 轮平均耗时、当前/目标步数和预计剩余分钟数。最后一轮在训练结束回调
中结算，checkpoint 和周期评估若在本轮触发，其开销会自然计入 ETA。恢复训练以模型的
实际 `num_timesteps` 为起点，目标步数为“起点 + 本次请求步数”。rollout 大小、PPO
epoch 数、平滑窗口和 ETA 单位同步写入 `run_metadata.json`。

验证：Python 语法与 CLI 帮助入口通过；2 个 `spawn` worker、每轮 64 环境步的两轮
smoke 正确输出两次开始和结束记录，最后一轮在训练结束时结算。随后从 128 步模型额外
续训 64 步，日志目标正确显示为 `192/192`；默认参数解析为 8,000,000，metadata
记录 rollout size、实际 PPO epoch 数、5 轮 ETA 窗口和分钟单位。`git diff --check`
通过。这些检查验证计时和恢复口径，不代表 800 万步正式训练结果。

## 39. Mobile 独立训练默认使用 12 个环境

面向 14 核训练服务器，`train_mobile_whisker_ppo.py` 的默认 `n_envs` 从 8 调整为 12，
为 PPO 主进程、单 worker 周期评估和系统进程保留计算余量。默认 `n_steps=512` 时每轮
采集 6,144 个环境步，能被默认 `batch_size=256` 整除。E4 formal 继续使用其独立配置的
8 个 worker，不随本次独立训练入口调整。

验证：默认参数解析得到 `timesteps=8,000,000`、`n_envs=12`、rollout size
`6,144`，批大小整除检查通过；Windows `spawn` 下以 12 个训练 worker 完成一轮
192-step smoke，终端记录 `vec_env_backend=subproc`，语法检查和
`git diff --check` 均通过。
