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
