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

当前代码主要完成了前两个方法的最小闭环：联合控制 DQN 与固定触须 DQN。

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

当前强化学习算法使用 Stable-Baselines3 的 DQN，底层神经网络由 PyTorch 实现。

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
6 个机器人移动动作 x 5 个触须采样动作 = 30 个离散动作
```

机器人移动动作包括前进、左转、右转、原地左旋、原地右旋和停止。

触须动作包括中心保持、窄幅扫描、宽幅扫描、左侧重点采样和右侧重点采样。

观测状态当前为 14 维，包含左右传感器读数、浓度差、浓度变化、左右 hit rate、触须角度、相对风向、机器人朝向和上一动作信息。

### 4.3 固定触须 baseline

`FixedWhiskerPlumeEnv` 是固定触须摆动 baseline。

该环境中智能体只控制机器人移动，触须按照固定策略周期性摆动。动作空间降为：

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

已实现两套 DQN 训练脚本：

```powershell
python scripts\train_fixed_whisker_dqn.py --timesteps 50000
python scripts\train_joint_dqn.py --timesteps 50000
```

默认模型输出：

```text
results/models/fixed_whisker_dqn.zip
results/models/joint_dqn.zip
```

默认日志输出：

```text
results/logs/fixed_whisker_dqn/
results/logs/joint_dqn/
```

已经做过 smoke training，证明训练流程可以运行。但 smoke training 只代表代码连通性测试，不能作为论文实验结论。

### 4.6 独立评估流程

评估逻辑已经从训练脚本中独立出来，统一放在：

```text
dual_whisker_rl/evaluation.py
```

评估脚本包括：

```powershell
python scripts\evaluate_fixed_whisker_dqn.py --episodes 50
python scripts\evaluate_joint_dqn.py --episodes 50
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

1. 跑正式 fixed-whisker DQN 与 joint DQN 对比实验。
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

python scripts\train_joint_dqn.py --timesteps 50000
python scripts\evaluate_joint_dqn.py --episodes 50
```

之后再做结果汇总和对照实验扩展。

## 8. 开发注意事项

- 修改环境动力学、奖励函数或观测空间后，应重新跑 fixed-whisker 和 joint 两套 smoke training。
- 修改评估指标后，应保证两个评估脚本输出字段一致。
- 修改气味羽流模型后，应重新生成气味场图和传感器响应图，检查是否仍符合任务直觉。
- 不要把 `results/` 中的训练结果、模型和图像提交到 Git。
- README 用于快速启动，本文档用于保存项目路线和阶段性判断。

