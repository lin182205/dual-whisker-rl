# 触须主动嗅觉项目可行性分析与未来规划（工程版）

> 版本：v1.0  
> 日期：2026-06-06  
> 适用对象：双触须主动嗅觉强化学习项目  
> 输入依据：现有 `PROJECT_RECORD.md` 项目记录、当前代码/硬件进展、实际工程落地经验  

---

## 0. 文档目的

本文档用于把当前“触须主动嗅觉项目”的研究目标、工程可行性、当前进度、主要风险和未来实施计划整理成一个可执行的项目路线图。它不是单纯的论文构想，也不是单纯的代码 TODO，而是面向后续真实开发的工程规划文档。

项目当前已经不再只是纯仿真问题，而是进入了“仿真训练 + STM32 舵机触须 + MQ-3 气体传感器 + Python 上位机”的软硬件联合阶段。因此，未来规划必须同时回答三个问题：

1. **这个方向是否可行？** 也就是主动触须是否有希望比固定双传感器获得更多有效气味信息。
2. **现在做到哪里了？** 也就是仿真、算法、硬件链路、数据记录、预处理分别处于什么成熟度。
3. **下一步怎么做最稳？** 也就是先验证什么，再训练什么，最后如何做 sim-to-real 和移动机器人找源。

本文档的核心结论是：

```text
该项目具备继续推进的可行性，但短期目标不应直接定义为“真实机器人端到端强化学习找源”。
更合理的工程路线是：
先验证双触须主动/扫描采样是否比固定传感器提供更高信息密度，
再做硬件可部署观测下的触须 PPO，
最后再把移动机器人控制重新加入。
```

---

## 1. 项目定位与研究主线

### 1.1 原始主线

项目原始目标是构建二维气味源定位强化学习环境，研究机器人移动控制与双触须主动嗅觉采样的联合优化。主方法可以概括为：

```text
双触须主动嗅觉采样 + 机器人移动联合控制
```

理想对比关系为：

```text
Joint RL: robot movement + whisker active sampling
>
Fixed-whisker RL: robot movement only, whiskers follow fixed scanning
>
Dual fixed electronic nose
>
Single fixed electronic nose
```

这个主线具有论文价值，因为它不是简单比较“有没有两个传感器”，而是比较“传感器采样位置是否可被主动控制”。如果能证明主动触须在间歇羽流中提高命中率、重新捕获能力或搜索效率，就能形成较清晰的研究贡献。

### 1.2 工程修正后的主线

结合当前 MQ-3 传感器和舵机硬件条件，项目主线应做一个工程化修正：

```text
短期重点：验证双触须主动采样能否提升局部气味信息获取能力。
中期重点：训练硬件可部署的触须策略，并在真实 MQ-3 数据上闭环执行。
长期重点：将主动触须采样重新接入移动机器人气味源搜索任务。
```

也就是说，短期不应强行要求系统输出精确气源方向，也不应要求 MQ-3 实时形成可靠浓度梯度。更现实的目标是验证主动触须能否回答：

```text
哪里更可能有气味？
哪里能产生响应变化？
哪里能制造左右不对称？
哪里值得继续采样？
```

这比“实时浓度梯度测量”更符合 MQ-3 慢响应、漂移和低成本硬件的实际特性。

---

## 2. 总体可行性结论

### 2.1 结论分级

| 目标 | 可行性判断 | 说明 |
|---|---:|---|
| 纯仿真环境中验证主动触须优于固定触须 | 高 | 环境、动作空间、评估、可视化已有基础，主要缺正式多 seed 实验。 |
| 固定位置真实双触须采样闭环 | 高 | PC-STM32-舵机-MQ-3-PC 链路已经具备最小闭环基础。 |
| 用 MQ-3 做精确实时空间浓度梯度测量 | 低到中 | MQ-3 慢响应、漂移、上电基线不一致，难以承担精密梯度测量。 |
| 用 MQ-3 验证主动采样是否提升信息获取 | 中到高 | 不依赖绝对浓度，只依赖命中率、趋势、左右差分、重新捕获等指标，更现实。 |
| 将仿真 PPO 直接迁移到真实触须硬件 | 中 | 需要观测对齐、预处理、历史堆叠和策略推理脚本；不能直接用仿真特权状态。 |
| 真实移动机器人端到端强化学习找源 | 中到低，适合作为后期目标 | 涉及气流场、底盘定位、传感器滞后、环境重复性和安全控制，短期不宜直接做。 |

### 2.2 一句话判断

```text
项目方向可行，但成功路径不是“先训练一个复杂 PPO 然后直接上硬件”，
而是“先把真实传感器数据变成稳定可解释的观测，再用公平 baseline 验证主动采样的增益”。
```

### 2.3 最值得保留的研究价值

本项目最值得保留的价值不是 MQ-3 传感器本身，也不是某个具体 PPO 网络，而是下面这个实验问题：

```text
在气味羽流间歇、传感器响应滞后、局部浓度不稳定的情况下，
主动改变双触须采样位置是否能比固定采样获得更高信息密度？
```

这可以通过一组可量化指标回答：

- 气味命中率是否提高；
- 平均有效响应是否提高；
- 响应趋势是否更明显；
- 左右不对称是否更强；
- 气味丢失后的重新捕获时间是否缩短；
- 在固定机器人位置时，主动采样是否优于最佳固定角度或周期扫描。

只要这些指标成立，即使暂时没有完整移动机器人找源，也可以形成一条稳健的阶段性成果线。

---

## 3. 当前进度分析

### 3.1 仿真环境进度

当前仿真侧已经完成了较完整的最小实验闭环，主要包括：

- 二维搜索区域；
- 固定气味源；
- 简化高斯气味羽流；
- 随机气味斑块和噪声；
- 差速机器人模型；
- 左右双触须采样点；
- 一阶慢响应气体传感器；
- Gymnasium 接口；
- Stable-Baselines3 训练流程；
- 训练、评估、指标统计和可视化输出。

主环境 `PlumeEnv` 支持机器人移动和触须联合控制，动作空间为：

```text
MultiDiscrete([6, 10, 10])
```

其中第一维是机器人动作，后两维分别是左右触须扇区。

固定触须 baseline `FixedWhiskerPlumeEnv` 已经实现，智能体只控制机器人移动，触须按固定周期扫描。该 baseline 很关键，因为它回答：

```text
如果触须只是固定摆动，RL 是否已经能完成基本找源？
```

从工程进度看，仿真环境已经超过“能跑通”的阶段，进入“需要正式实验、统一评价、避免结论偶然性”的阶段。

### 3.2 强化学习算法进度

当前算法侧已经有两条线：

```text
固定触须 baseline：DQN
联合控制主方法：PPO
```

选择 PPO 做联合控制是合理的，因为联合控制动作空间是 `MultiDiscrete([6, 10, 10])`，而 DQN 更适合单一离散动作。当前还没有必要引入更复杂的算法，例如 SAC、TD3、RNN-PPO 或 Transformer policy。

当前更重要的是：

1. 固定评估协议；
2. 多随机种子训练；
3. baseline 公平；
4. 训练曲线与最终指标一致；
5. 不让策略利用仿真特权状态。

因此算法侧的成熟度可以判断为：

```text
代码闭环：已完成
正式对比实验：未完成
可论文引用结论：未形成
sim-to-real 可部署策略：尚未完成
```

### 3.3 评估与可视化进度

当前已经有统一评估模块 `dual_whisker_rl/evaluation.py`，指标包括：

- `success_rate`；
- `mean_return`；
- `mean_steps`；
- `mean_final_distance`；
- `mean_path_length`；
- `mean_odor_hits`；
- `mean_reacquisition_time`；
- `reacquisition_events`。

可视化能力包括：

- 气味羽流浓度场；
- 机器人轨迹；
- 左右传感器响应曲线；
- 触须角度曲线；
- 评估轨迹 GIF 动画。

这说明项目已经具备“实验可解释性”的基础，但仍缺少自动化汇总工具，例如：

```text
scripts/compare_results.py
scripts/plot_training_curves.py
scripts/summarize_multi_seed_results.py
```

没有这些工具，后续实验很容易停留在单次训练截图，难以形成可信结论。

### 3.4 硬件链路进度

硬件平台已经明确：

```text
控制板：STM32F103C8T6
执行机构：两个舵机
传感器：两个 MQ-3 气体传感器
上位机：PC Python
下位机：STM32 C / HAL
```

当前引脚规划为：

```text
左舵机 PWM 信号  -> PA0 / TIM2_CH1
右舵机 PWM 信号  -> PA1 / TIM2_CH2
左 MQ-3 AO       -> PA2 / ADC1_IN2
右 MQ-3 AO       -> PA3 / ADC1_IN3
USART1_TX        -> PA9
USART1_RX        -> PA10
```

当前串口协议为：

```text
PC -> STM32:
STEP left_sector right_sector

STM32 -> PC:
{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}
```

这一设计非常适合当前阶段，因为它把上位机策略和下位机实时控制解耦：

- Python 负责策略、记录、分析、绘图；
- STM32 负责 PWM、ADC、串口和低层稳定执行。

这种分工符合实际机器人系统开发习惯，也便于调试。

### 3.5 触须专用仿真环境进度

已经新增 `WhiskerOnlyPuffEnv`，用于固定机器人基座、只训练左右触须扇区选择。其动作空间为：

```text
MultiDiscrete([10, 10])
```

这与硬件串口命令天然对应：

```text
[left_sector, right_sector] -> STEP left_sector right_sector
```

该环境具有几个重要优点：

1. 机器人不移动，问题更单纯；
2. 场地为 1m x 1m，触须长度 15cm，贴近硬件尺度；
3. 加入了动态 puff 羽流，而不是只用静态高斯场；
4. 加入了舵机速度约束，避免仿真中触须瞬间跳转；
5. 支持历史观测堆叠，适合 MQ-3 慢响应特性。

这条线非常重要。它是从“完整机器人找源”退一步，先做“固定位置主动触须采样”的中间层。工程上这是正确选择，因为它能隔离移动底盘、定位误差和复杂环境变量。

### 3.6 传感器预处理进度

真实 MQ-3 测试已经发现：

```text
洁净空气中 ADC 仍会缓慢下降；
每次重新上电后的基线不一致；
原始 ADC 不适合直接作为 PPO 输入。
```

针对这个问题，已经新增：

```text
dual_whisker_rl/hardware/sensor_preprocess.py
```

其核心思想是：

```text
raw ADC
-> baseline subtraction
-> EMA smoothing
-> trend / response feature
-> normalization
-> PPO observation
```

当前特征包括：

```text
norm_signal
norm_smooth
norm_trend
norm_estimated_input
norm_signal_diff
norm_smooth_diff
norm_trend_diff
norm_estimated_input_diff
```

这是项目从“能采到数据”走向“能用数据”的关键一步。没有预处理，真实 MQ-3 的漂移会直接破坏策略输入；有了预处理，后续才有机会做 sim-to-real。

---

## 4. 工程经验视角下的主要风险

### 4.1 风险一：MQ-3 慢响应与基线漂移

#### 表现

- 原始 ADC 随时间缓慢变化；
- 每次上电后的基线不同；
- 当前读数可能反映前几个扇区的气味历史，而不是当前扇区；
- 触须快速摆动时会产生响应混叠。

#### 影响

如果直接把 `left_adc/right_adc` 输入 PPO，策略可能学到的是传感器漂移、上电状态或历史残留，而不是真正的空间气味信息。

#### 应对策略

- 预热后再采集；
- 每次实验记录基线段；
- 使用 `raw - baseline` 而不是 raw；
- 使用 EMA、trend、estimated_input；
- 控制触须 dwell time，不要无意义快速乱扫；
- 使用历史观测堆叠；
- 在评估中单独记录原始 ADC 和预处理特征。

### 4.2 风险二：触须动作快于传感器动态

舵机可以在几百毫秒内切换扇区，但 MQ-3 的气体响应通常慢得多。如果触须每 0.2s 或 0.5s 切换一次扇区，传感器输出可能仍主要来自上一位置或更早位置。

#### 应对策略

- 在硬件实验中显式比较不同 `settle_ms`：300ms、500ms、1000ms、2000ms；
- 在数据中记录 `sector`、`t_pc_s`、后续增加 `t_mcu_ms`；
- 增加 `dwell_step` 或 `time_since_sector_change` 作为观测；
- 训练时加入舵机速度限制和传感器慢响应；
- 不把“瞬时左右差分”作为唯一指标，而是看趋势和重新捕获能力。

### 4.3 风险三：气流实验不可重复

真实气味羽流受气流、温湿度、房间扰动、气源挥发、传感器位置影响很大。如果没有基本实验规范，硬件数据会很难解释。

#### 应对策略

- 固定风扇位置、气源位置、触须基座位置；
- 每次实验记录气源类型、距离、风扇档位、环境备注；
- 做同一模式多次重复；
- 先比较同一天、同一环境下的 fixed / periodic / random；
- 不跨天直接比较绝对 ADC，只比较归一化指标。

### 4.4 风险四：baseline 不公平

主动触须如果只和一个很差的固定角度比较，结论会很弱。工程和论文中都必须避免这种问题。

#### 应对策略

固定传感器 baseline 至少应包含多个角度：

```text
fixed_30deg
fixed_60deg
fixed_90deg
fixed_120deg
fixed_150deg
```

主动策略应至少与以下模式比较：

```text
best fixed angle
periodic scan
random scan
heuristic active scan
PPO active scan
```

只有当主动策略优于最佳固定角度，或在重新捕获、信息密度等指标上明显更好，结论才有说服力。

### 4.5 风险五：仿真特权状态导致 sim-to-real 失败

当前触须仿真环境中仍可能包含风向、气源距离等真实硬件不易获得的信息。如果 PPO 依赖这些状态，迁移到硬件时会失效。

#### 应对策略

逐步将触须 PPO 观测改为硬件可获得特征：

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
time_since_change_norm
```

仿真和硬件共用同一套 observation builder，减少训练/部署不一致。

### 4.6 风险六：串口与数据完整性问题

串口文本协议简单好调，但长期运行时可能出现超时、半包、乱码、重复返回、异常 JSON 等问题。

#### 应对策略

- 每条命令和返回都记录日志；
- 返回中增加 `seq`、`t_mcu_ms`、`status`；
- 上位机对 JSON 解析失败做重试；
- 保存 raw serial line；
- 超时后不要继续盲目训练或执行策略；
- 硬件 rollout 脚本中加入安全停止命令。

---

## 5. 项目阶段成熟度判断

| 模块 | 当前状态 | 成熟度 | 下一步重点 |
|---|---|---:|---|
| 二维仿真环境 | 已闭环 | 高 | 固定配置、补多 seed、补对比表。 |
| 联合控制 PPO | 已能训练 | 中 | 正式训练评估，避免单次 smoke training 当结论。 |
| 固定触须 DQN baseline | 已能训练 | 中 | 与 PPO 同协议评估。 |
| 触须专用环境 | 已建立 | 中 | 改成硬件可部署观测，做 PPO 对比。 |
| 动态 puff 羽流 | 已加入 | 中 | 调参与真实数据相符，避免过于理想化。 |
| 传感器一阶模型 | 已有 | 中 | 根据 MQ-3 数据拟合响应/恢复参数。 |
| STM32 固件 | 最小闭环代码已有 | 中 | CubeIDE 编译烧录验证，增加 seq/t_mcu_ms/status。 |
| Python 串口客户端 | 已准备 | 中到高 | 强化异常处理和数据记录。 |
| MQ-3 数据预处理 | 模块已新增 | 中 | 用真实 baseline 数据验证参数。 |
| 硬件采样模式对比 | 尚未完成 | 低 | 新增 `compare_hardware_sampling_modes.py`。 |
| 硬件 PPO 推理 | 尚未完成 | 低 | 等预处理和观测对齐后再做。 |
| 移动机器人接入 | 未开始 | 低 | 放到主动触须验证之后。 |

---

## 6. 未来总体路线

未来路线建议分为六个阶段：

```text
阶段 1：传感器数据质量验证
阶段 2：硬件采样模式对比
阶段 3：仿真观测硬件化
阶段 4：触须 PPO 训练与离线评估
阶段 5：真实硬件触须策略推理
阶段 6：重新接入移动机器人找源
```

这一路线的关键思想是：

```text
先证明真实传感器数据可用，
再证明扫描触须比固定传感器有信息优势，
再证明主动策略比简单扫描更好，
最后再做完整机器人搜索。
```

---

## 7. 阶段 1：传感器数据质量验证

### 7.1 目标

验证真实 MQ-3 数据经过预处理后，是否能形成稳定、可解释、可用于 PPO 的观测特征。

该阶段不训练 PPO，不比较复杂策略，只回答一个基础问题：

```text
当前 MQ-3 原始数据经过 baseline subtraction、EMA、trend、estimated_input 后，
能否把洁净空气漂移压低，并保留真实气体刺激响应？
```

### 7.2 需要新增或完善的脚本

```text
scripts/analyze_hardware_baseline.py
scripts/hardware_live_plot.py
```

### 7.3 `analyze_hardware_baseline.py` 功能要求

输入：

```text
results/hardware/live_plot.csv
results/hardware/smoke_test.csv
results/hardware/sector_scan.csv
```

至少支持字段：

```text
step,t_pc_s,left_sector,right_sector,left_adc,right_adc
```

输出：

```text
results/hardware/baseline_analysis.json
results/hardware/baseline_preprocess.png
results/hardware/baseline_summary.md
```

统计指标：

```text
left_mean_adc
right_mean_adc
left_std_adc
right_std_adc
left_min_adc
right_min_adc
left_max_adc
right_max_adc
left_drift_slope_adc_per_s
right_drift_slope_adc_per_s
left_first_last_delta
right_first_last_delta
left_noise_std_after_detrend
right_noise_std_after_detrend
```

预处理曲线：

```text
raw_adc
baseline
signal = raw_adc - baseline
smooth
trend
estimated_input
```

### 7.4 验收标准

阶段 1 完成的标志不是“图画出来了”，而是满足以下条件：

- 能明确看出 raw ADC 的漂移幅度；
- 能明确看出预处理后 signal/smooth/trend 是否稳定；
- 能判断当前参数是否把洁净空气下的漂移压到可接受范围；
- 能在气体刺激实验中看到 smooth/trend 的可解释变化；
- 能形成一份 `baseline_summary.md`，记录该次实验是否适合后续训练/对比。

### 7.5 决策规则

```text
如果洁净空气下预处理特征仍然大幅漂移：
    暂停 PPO，先调 baseline 更新、EMA、归一化窗口和传感器预热。

如果预处理后趋势特征稳定，但气体刺激响应太弱：
    调整气源距离、风扇、触须位置、采样停留时间。

如果预处理后洁净空气稳定，气体刺激时有明显响应：
    进入阶段 2，做采样模式对比。
```

---

## 8. 阶段 2：硬件采样模式对比

### 8.1 目标

在真实硬件上比较固定传感器、周期扫描、随机扫描三类策略的信息获取能力。

这个阶段非常关键，因为它决定后续是否值得投入 PPO 主动策略。

核心问题：

```text
在真实 MQ-3 + 舵机触须硬件上，
触须扫描是否比固定角度传感器获得更多有效气味信息？
```

### 8.2 需要新增脚本

```text
scripts/compare_hardware_sampling_modes.py
```

### 8.3 支持的模式

```text
fixed_angle
periodic_scan
random_scan
```

后续扩展：

```text
heuristic_active
ppo_active
```

### 8.4 fixed_angle 设计

不能只设置一个固定角度。建议固定角度至少包括：

```text
30 deg
60 deg
90 deg
120 deg
150 deg
```

对应扇区可以近似映射为：

```text
sector 1 -> 27 deg
sector 3 -> 63 deg
sector 5 -> 99 deg
sector 7 -> 135 deg
sector 8 -> 153 deg
```

左右触须可先对称固定，例如：

```text
left_sector = k
right_sector = k
```

后续可以再加入左右不对称固定组合。

### 8.5 periodic_scan 设计

基础扫描：

```text
0 -> 1 -> 2 -> ... -> 9 -> 8 -> ... -> 1 -> 0
```

往返扫描比单向跳回更平滑，减少舵机大角度快速回摆。

建议记录：

```text
scan_period_steps
settle_ms
dwell_steps
```

### 8.6 random_scan 设计

随机扫描用于判断主动或周期策略是否只是因为“采样位置更多”而提升。

应避免每一步完全随机大跳，可以比较两种 random：

```text
random_any_sector：每步随机选 0-9
random_neighbor：每步只向相邻扇区移动或停留
```

第二种更符合舵机机械限制，也更公平。

### 8.7 输出数据

每次运行保存：

```text
results/hardware/sampling_modes/{run_id}/raw.csv
results/hardware/sampling_modes/{run_id}/features.csv
results/hardware/sampling_modes/{run_id}/metrics.json
results/hardware/sampling_modes/{run_id}/summary.md
results/hardware/sampling_modes/{run_id}/plots.png
results/hardware/sampling_modes/{run_id}/config.yaml
```

`raw.csv` 字段：

```text
step
seq
t_pc_s
t_mcu_ms
mode
left_sector
right_sector
left_adc
right_adc
raw_serial_line
```

`features.csv` 字段：

```text
step
mode
left_sector
right_sector
left_signal
right_signal
left_smooth
right_smooth
left_trend
right_trend
left_estimated_input
right_estimated_input
smooth_diff
trend_diff
max_smooth
abs_smooth_diff
```

### 8.8 信息获取指标

建议统一使用以下指标：

```text
odor_hit_rate
mean_max_smooth
mean_abs_smooth_diff
mean_abs_trend_sum
positive_trend_rate
reacquisition_count
mean_reacquisition_steps
high_information_rate
sector_entropy
sector_coverage
```

定义建议：

```text
odor_hit_rate = mean(max(left_smooth, right_smooth) > threshold)
mean_max_smooth = mean(max(left_smooth, right_smooth))
mean_abs_smooth_diff = mean(abs(left_smooth - right_smooth))
mean_abs_trend_sum = mean(abs(left_trend) + abs(right_trend))
positive_trend_rate = mean(max(left_trend, right_trend) > trend_threshold)
high_information_rate = mean((max_smooth > threshold) or (abs_smooth_diff > diff_threshold) or (abs_trend_sum > trend_threshold))
sector_entropy = entropy(distribution of visited sectors)
sector_coverage = number of visited sectors / 10
```

### 8.9 验收标准

进入下一阶段前，至少要有以下结论之一：

```text
结论 A：periodic_scan 在命中率、趋势或重新捕获上优于多数 fixed_angle。
结论 B：random_neighbor 能提升覆盖，但不如 periodic_scan 稳定。
结论 C：所有扫描都不如最佳 fixed_angle，说明硬件采样节奏或传感器布置需要调整。
```

如果出现结论 C，不应急着训练 PPO，而应优先检查：

- 触须是否真的把传感器移动到了不同气味区域；
- 舵机停留时间是否太短；
- 气源/风扇布置是否让空间差异太弱；
- MQ-3 是否饱和或响应太慢；
- 预处理是否把真实响应也过滤掉了。

---

## 9. 阶段 3：仿真观测硬件化

### 9.1 目标

让仿真训练时 PPO 看到的观测，尽量接近真实硬件部署时能构造的观测。

当前应减少或移除以下仿真特权信息：

```text
真实气源距离
真实风向
真实浓度场内部状态
直接 raw concentration
```

保留或新增硬件可获得信息：

```text
左右预处理后的传感器特征
左右触须角度或扇区
上一动作
动作停留时间
历史观测窗口
```

### 9.2 建议新增模块

```text
dual_whisker_rl/observation/
├── __init__.py
├── whisker_observation_builder.py
└── feature_normalizer.py
```

或者先简单放在：

```text
dual_whisker_rl/hardware/observation_builder.py
```

### 9.3 统一观测格式

建议定义硬件可部署触须观测为单帧 12 到 16 维：

```text
left_signal
right_signal
left_smooth
right_smooth
left_trend
right_trend
left_estimated_input
right_estimated_input
smooth_diff
trend_diff
left_angle_norm
right_angle_norm
left_sector_norm
right_sector_norm
time_since_left_change_norm
time_since_right_change_norm
```

如果先简化，可用 12 维：

```text
left_signal
right_signal
left_smooth
right_smooth
left_trend
right_trend
smooth_diff
trend_diff
left_angle_norm
right_angle_norm
left_sector_norm
right_sector_norm
```

然后使用历史堆叠：

```text
history_length = 10 / 20 / 30
```

当前触须环境 `dt = 0.2s` 时：

```text
history_length = 10 -> 约 2 秒历史
history_length = 20 -> 约 4 秒历史
history_length = 30 -> 约 6 秒历史
```

考虑 MQ-3 慢响应，建议优先比较：

```text
history_length = 1, 10, 20, 30
```

### 9.4 仿真传感器模型调整

仿真中应引入与硬件更接近的扰动：

```text
baseline offset randomization
baseline drift
sensor noise
asymmetric response/recovery
ADC quantization
response lag
```

但不要一次加入所有复杂性。建议顺序为：

```text
1. baseline offset + Gaussian noise
2. slow drift
3. response/recovery asymmetry
4. ADC quantization
5. domain randomization
```

每加入一个复杂性，都要确认 periodic_scan、random_scan 和 PPO 的相对关系是否仍可解释。

### 9.5 验收标准

阶段 3 完成的标志：

- 仿真和硬件使用同一套 observation builder；
- PPO 观测中不包含硬件不可获得的特权状态；
- 观测维度、归一化范围、历史堆叠长度在 metadata 中记录；
- 可用随机策略跑仿真并导出与硬件 `features.csv` 结构接近的数据；
- 后续硬件 policy rollout 不需要临时拼凑观测。

---

## 10. 阶段 4：触须 PPO 训练与离线评估

### 10.1 目标

在硬件可部署观测下，训练触须专用 PPO，并在仿真中与固定、周期、随机、启发式策略比较。

### 10.2 训练命令建议

基础命令：

```powershell
python scripts\train_whisker_only_ppo.py --timesteps 50000 --n-envs 4 --history-length 20
```

正式实验不要只跑一次。建议用多 seed：

```powershell
python scripts\train_whisker_only_ppo.py --timesteps 200000 --n-envs 8 --history-length 20 --seed 1
python scripts\train_whisker_only_ppo.py --timesteps 200000 --n-envs 8 --history-length 20 --seed 2
python scripts\train_whisker_only_ppo.py --timesteps 200000 --n-envs 8 --history-length 20 --seed 3
python scripts\train_whisker_only_ppo.py --timesteps 200000 --n-envs 8 --history-length 20 --seed 4
python scripts\train_whisker_only_ppo.py --timesteps 200000 --n-envs 8 --history-length 20 --seed 5
```

具体步数可以根据训练曲线调整，但正式结论必须来自多 seed，而不是单个模型。

### 10.3 需要新增评估脚本

```text
scripts/evaluate_whisker_sampling_policies.py
scripts/compare_whisker_only_results.py
```

支持策略：

```text
fixed_angle
periodic_scan
random_scan
heuristic_active
ppo_active
```

### 10.4 启发式 active baseline

在 PPO 前应有一个简单启发式策略，避免论文只比较神经网络与弱 baseline。

可设计为：

```text
如果最近 max_smooth 低：扩大扫描范围；
如果某侧 trend 为正：该侧在附近扇区细扫；
如果左右差分大：保持高响应侧，另一侧探索；
如果长时间无 hit：执行周期性全局扫描。
```

启发式策略的意义是：

- 检查奖励和指标是否合理；
- 提供可解释 baseline；
- 判断 PPO 是否真正学到了比规则更好的策略。

### 10.5 PPO 奖励建议

触须专用 PPO 的 reward 不宜只奖励绝对浓度。建议由以下部分组成：

```text
r_hit：采样到有效气味
r_trend：传感器响应上升
r_contrast：左右产生不对称
r_reacquire：气味丢失后重新捕获
r_coverage：避免长期停在单一扇区
r_motion_cost：惩罚过于频繁大幅摆动
r_time_cost：轻微时间惩罚
```

示例形式：

```text
reward =
    w_hit * hit_bonus
  + w_trend * max(0, max(left_trend, right_trend))
  + w_contrast * abs(left_smooth - right_smooth)
  + w_reacquire * reacquire_bonus
  + w_coverage * local_exploration_bonus
  - w_motion * sector_change_magnitude
  - w_time
```

注意：

```text
不要让 r_coverage 过强，否则策略会变成无意义扫动；
不要让 r_contrast 过强，否则策略可能故意制造噪声差分；
不要只奖励 hit，否则策略可能停在一个偶然高响应扇区不动。
```

### 10.6 离线评估指标

仿真评估指标应尽量与硬件阶段一致：

```text
odor_hit_rate
mean_max_smooth
mean_abs_smooth_diff
mean_abs_trend_sum
positive_trend_rate
mean_reacquisition_steps
high_information_rate
sector_entropy
motion_cost
```

额外记录：

```text
mean_return
policy_sector_distribution
left_right_sector_correlation
time_in_high_response_region
```

### 10.7 验收标准

PPO 进入硬件部署前，应满足：

- 多 seed 平均性能优于 random_scan；
- 至少在部分关键指标上接近或优于 periodic_scan；
- 不依赖仿真特权状态；
- 扇区选择分布不是完全塌缩到单一扇区；
- 可视化轨迹能解释策略行为；
- 对不同 plume seed 和 sensor noise 有一定鲁棒性。

如果 PPO 不能优于周期扫描，仍然可以保留周期扫描作为工程有效方案，并把 PPO 作为探索性结果，而不是强行包装成主结论。

---

## 11. 阶段 5：真实硬件触须策略推理

### 11.1 目标

将训练好的触须 PPO 接入真实 STM32 舵机触须系统，实现：

```text
真实 MQ-3 ADC
-> Python 预处理
-> 历史观测构造
-> PPO 输出 left_sector/right_sector
-> 串口 STEP 命令
-> STM32 控制舵机并返回新 ADC
-> 保存日志和图像
```

### 11.2 需要新增脚本

```text
scripts/hardware_whisker_policy_rollout.py
```

### 11.3 基本流程

```text
1. 打开串口；
2. 加载 PPO 模型；
3. 初始化 DualGasPreprocessor；
4. 初始化 observation history buffer；
5. 执行若干步 warmup 或 baseline 采集；
6. 构造第一帧观测；
7. PPO predict 输出动作；
8. 发送 STEP left_sector right_sector；
9. 接收 STM32 JSON；
10. 更新预处理特征和历史观测；
11. 记录 raw.csv / features.csv / actions.csv；
12. 实时绘图或保存动态图；
13. 异常时安全停止。
```

### 11.4 运行命令示例

```powershell
python scripts\hardware_whisker_policy_rollout.py ^
  --port COM3 ^
  --model results\models\whisker_only_ppo.zip ^
  --steps 300 ^
  --history-length 20 ^
  --deterministic ^
  --output-dir results\hardware\policy_rollout
```

### 11.5 安全与异常处理

必须处理：

```text
串口打开失败；
STM32 无返回；
JSON 解析失败；
返回 sector 与发送 sector 不一致；
ADC 超范围；
连续超时；
用户 Ctrl+C；
模型输出非法动作；
```

异常时建议：

```text
1. 停止发送新动作；
2. 可选发送 CENTER 或 STEP 5 5；
3. 保存已有日志；
4. 输出错误摘要。
```

### 11.6 硬件 PPO 不应直接追求“找源”

第一版硬件 PPO 只看固定位置触须采样表现，指标仍然是信息获取：

```text
odor_hit_rate
mean_max_smooth
mean_abs_smooth_diff
reacquisition_time
high_information_rate
motion_cost
```

硬件 PPO 与以下模式比较：

```text
best_fixed_angle
periodic_scan
random_neighbor
heuristic_active
ppo_active
```

### 11.7 验收标准

硬件策略推理阶段成功的标志是：

- PPO 能稳定运行完整 episode，不频繁串口异常；
- PPO 输出动作可解释，不是随机抖动；
- PPO 的信息获取指标至少优于 random_neighbor；
- 如果不优于 periodic_scan，也能通过可视化解释失败原因；
- 所有实验都有完整 CSV、JSON、图像和配置记录。

---

## 12. 阶段 6：重新接入移动机器人找源

### 12.1 进入条件

只有在以下条件基本满足后，才建议重新接入移动机器人：

```text
1. MQ-3 预处理稳定；
2. 固定位置触须扫描优于固定传感器；
3. PPO 或启发式主动触须在至少部分指标上优于 random；
4. 硬件触须策略推理可以稳定运行；
5. 数据记录和评估脚本已经标准化。
```

### 12.2 移动机器人阶段目标

移动机器人阶段不应一上来做真实硬件端到端 RL。建议按以下顺序：

```text
1. 仿真中恢复 robot movement + active whisker joint control；
2. 用硬件化观测替代仿真特权状态；
3. 与 fixed-whisker、dual-fixed-nose、single-nose 做多 seed 对比；
4. 在真实机器人上先执行规则移动 + 主动触须；
5. 再考虑加载仿真策略做低速、安全、有限区域测试。
```

### 12.3 真实机器人初期策略

真实硬件初期可采用规则移动，不必马上端到端 PPO：

```text
如果有气味 hit：低速前进 + 小幅横向搜索；
如果左右差分明显：向高响应侧微调；
如果长时间无 hit：横风扫描或螺旋搜索；
主动触须持续执行 PPO/周期扫描，用于提供局部信息。
```

这种分层控制更符合工程实际：

```text
触须策略负责信息获取；
移动策略负责安全搜索；
后续再考虑联合优化。
```

### 12.4 最终论文实验结构

长期论文结构可以设计为：

```text
实验 1：仿真中主动触须采样有效性
实验 2：真实 MQ-3 硬件中扫描触须相对固定传感器的信息增益
实验 3：硬件可部署观测下 PPO 主动触须策略对比
实验 4：仿真中机器人移动 + 主动触须联合找源
实验 5：真实移动平台上的概念验证
```

这样即使真实移动平台结果不完美，前 1-3 个实验也能支撑项目核心贡献。

---

## 13. 建议新增文件与目录

### 13.1 硬件分析与对比

```text
scripts/
├── analyze_hardware_baseline.py
├── compare_hardware_sampling_modes.py
├── hardware_live_plot.py
├── hardware_whisker_policy_rollout.py
└── summarize_hardware_runs.py
```

### 13.2 仿真评估与汇总

```text
scripts/
├── evaluate_whisker_sampling_policies.py
├── compare_whisker_only_results.py
├── summarize_multi_seed_results.py
├── plot_training_curves.py
└── make_paper_figures.py
```

### 13.3 观测构造与配置

```text
dual_whisker_rl/
├── observation/
│   ├── __init__.py
│   ├── whisker_observation_builder.py
│   └── feature_normalizer.py
└── hardware/
    ├── sensor_preprocess.py
    ├── serial_client.py
    ├── hardware_logger.py
    └── safety.py
```

### 13.4 结果目录

```text
results/
├── hardware/
│   ├── baseline_analysis/
│   ├── sampling_modes/
│   └── policy_rollout/
├── models/
├── logs/
├── tensorboard/
├── figures/
└── summaries/
```

---

## 14. 数据记录规范

### 14.1 每次实验必须保存的文件

```text
config.yaml
run_metadata.json
raw.csv
features.csv
metrics.json
summary.md
plots.png
```

### 14.2 `run_metadata.json` 建议字段

```json
{
  "run_id": "20260606_001_periodic_scan",
  "mode": "periodic_scan",
  "created_at": "2026-06-06T00:00:00",
  "git_commit": "unknown",
  "hardware": {
    "mcu": "STM32F103C8T6",
    "sensors": "MQ-3 x2",
    "actuators": "servo x2",
    "port": "COM3"
  },
  "protocol": {
    "command": "STEP left_sector right_sector",
    "baudrate": 115200,
    "timeout_s": 2.0
  },
  "sampling": {
    "settle_ms": 500,
    "delay_s": 0.2,
    "cycles": 5
  },
  "preprocess": {
    "baseline_alpha": 0.001,
    "smooth_alpha": 0.1,
    "trend_window": 5
  },
  "environment_notes": {
    "source_type": "alcohol",
    "fan": "low",
    "distance_cm": 30,
    "notes": ""
  }
}
```

### 14.3 为什么 metadata 很重要

气味实验高度依赖环境。如果没有 metadata，后续无法判断结果差异来自策略，还是来自风扇、气源、传感器预热、串口间隔或当天环境变化。

工程上建议把实验结果当成“不可复现风险很高的数据”，必须每次记录配置。

---

## 15. 实验决策树

```text
开始
 |
 |-- 阶段 1：洁净空气 baseline 是否稳定？
 |       |-- 否：调预热、baseline、EMA、硬件供电、接线
 |       |-- 是：进入气体刺激测试
 |
 |-- 气体刺激下预处理特征是否有可解释响应？
 |       |-- 否：调气源距离、风扇、传感器位置、停留时间
 |       |-- 是：进入采样模式对比
 |
 |-- periodic_scan 是否优于多数 fixed_angle？
 |       |-- 否：先不要 PPO，优化硬件采样节奏和实验布置
 |       |-- 是：说明主动/扫描触须有潜力
 |
 |-- random_scan 是否接近 periodic_scan？
 |       |-- 是：说明主要收益来自覆盖范围，PPO 需证明更高效
 |       |-- 否：说明扫描结构和时序很重要
 |
 |-- 仿真硬件化观测 PPO 是否优于 random？
 |       |-- 否：改 reward、观测、历史长度、羽流随机化
 |       |-- 是：进入硬件 policy rollout
 |
 |-- 硬件 PPO 是否优于 random 或接近 periodic？
 |       |-- 否：分析 sim-to-real 差异，保留周期扫描为工程方案
 |       |-- 是：进入移动机器人扩展
```

---

## 16. 可作为论文或报告的阶段性结论模板

### 16.1 如果周期扫描优于固定传感器

可以形成如下结论：

```text
在低成本 MQ-3 传感器存在慢响应和基线漂移的条件下，
主动/周期性改变采样位置仍能提高局部气味信息获取能力。
这说明双触须结构的价值不在于精确瞬时浓度梯度，
而在于扩大局部采样范围、提高命中率和重新捕获能力。
```

### 16.2 如果 PPO 优于周期扫描

可以形成更强结论：

```text
基于历史传感器响应的主动触须策略，
可以在动态间歇羽流中选择更有信息量的采样扇区，
相较固定扫描进一步提高高信息采样比例或缩短重新捕获时间。
```

### 16.3 如果 PPO 没有优于周期扫描

也不是完全失败，可以形成工程结论：

```text
在当前 MQ-3 响应速度和采样节奏下，
规则周期扫描已能获得大部分主动采样收益，
复杂强化学习策略的优势受传感器滞后和真实羽流噪声限制。
后续应优先改进传感器动态建模、停留时间和气流实验重复性。
```

这样的结论比强行声称 PPO 成功更可信。

---

## 17. 近期优先任务清单

### P0：必须优先完成

- [ ] 用 `analyze_hardware_baseline.py` 验证 MQ-3 洁净空气漂移和预处理效果；
- [ ] 在 `hardware_live_plot.py` 中实时显示 raw、signal、smooth、trend、diff；
- [ ] 增加 STM32 返回字段：`seq`、`t_mcu_ms`、`status`；
- [ ] 完成 `compare_hardware_sampling_modes.py` 的 fixed / periodic / random 对比；
- [ ] 固定实验数据格式：raw.csv、features.csv、metrics.json、summary.md。

### P1：完成 P0 后推进

- [ ] 统一仿真和硬件 observation builder；
- [ ] 移除触须 PPO 观测中的硬件不可获得状态；
- [ ] 训练硬件化观测的 `whisker_only_ppo`；
- [ ] 增加 `evaluate_whisker_sampling_policies.py`；
- [ ] 多 seed 汇总 fixed / periodic / random / PPO。

### P2：中期扩展

- [ ] 编写 `hardware_whisker_policy_rollout.py`；
- [ ] 在真实硬件上运行 PPO 触须策略；
- [ ] 比较 PPO 与 best fixed、periodic、random；
- [ ] 根据真实数据拟合传感器响应和漂移参数；
- [ ] 做 sim-to-real 差异分析。

### P3：后期扩展

- [ ] 加入双固定电子鼻 baseline；
- [ ] 加入单电子鼻 baseline；
- [ ] 重新训练 robot + whisker joint PPO；
- [ ] 接入移动底盘规则控制；
- [ ] 最后再尝试真实移动机器人端到端策略。

---

## 18. 最终建议

### 18.1 目前不要做的事

短期内不建议：

```text
1. 直接把完整机器人 PPO 上真实硬件；
2. 在 MQ-3 原始 ADC 上直接训练策略；
3. 只跑一次训练就写实验结论；
4. 只和一个固定角度 baseline 比较；
5. 一次性加入复杂传感器模型、复杂神经网络和移动底盘；
6. 在没有风扇/气源/预热记录的情况下比较不同天的数据。
```

### 18.2 目前最应该做的事

短期最应该做：

```text
1. 把 MQ-3 真实数据预处理做好；
2. 用真实硬件比较 fixed / periodic / random；
3. 确认扫描触须是否真的带来信息增益；
4. 把仿真 PPO 观测改成硬件可部署形式；
5. 再训练和部署主动触须策略。
```

### 18.3 项目成败的关键

这个项目真正的关键不是 PPO 算法本身，而是：

```text
能否定义一个公平、可重复、能体现主动采样价值的实验问题。
```

如果实验问题定义为“MQ-3 精确测量瞬时浓度梯度”，项目风险很高；如果定义为“主动触须提高局部气味信息获取能力”，项目可行性明显更高，也更符合当前硬件现实。

因此，建议将下一阶段项目目标正式写为：

```text
在固定触须基座和真实 MQ-3 传感器条件下，
比较固定角度、周期扫描、随机扫描和主动策略的气味信息获取能力，
验证双触须主动采样是否相较固定传感器提供更高命中率、响应趋势、左右不对称和重新捕获能力。
```

这是当前最稳、最容易落地、也最能支撑后续论文主线的路线。

---

## 19. 参考资料与项目依据

### 19.1 项目内部依据

- `PROJECT_RECORD.md`：当前项目目标、代码结构、仿真环境、硬件链路、传感器预处理和短期验证计划。

### 19.2 外部工程参考

- STMicroelectronics：STM32F103C8 产品页与数据手册，用于确认供电范围、ADC、定时器、USART 等外设能力。
- Stable-Baselines3 文档：PPO 支持 `MultiDiscrete` 动作空间，DQN 更适合单一 `Discrete` 动作空间。
- MQ-3 气体传感器数据手册：MQ-3 为带加热器的 SnO2 气敏传感器，实际工程中需关注预热、漂移、慢响应和环境影响。

---

## 20. 附录：建议的 `compare_hardware_sampling_modes.py` 参数

```text
--port COM3
--mode fixed_angle / periodic_scan / random_scan
--fixed-sector 5
--cycles 5
--steps 300
--settle-ms 500
--delay-s 0.2
--baudrate 115200
--output-dir results/hardware/sampling_modes
--preprocess-config configs/hardware.yaml
--source-note "alcohol source, fan low, distance 30cm"
```

---

## 21. 附录：硬件策略 rollout 伪代码

```python
client = SerialClient(port="COM3")
pre = DualGasPreprocessor(config)
policy = PPO.load(model_path)
history = ObservationHistory(length=20)

# baseline warmup
for _ in range(warmup_steps):
    data = client.step(5, 5)
    features = pre.update(data.left_adc, data.right_adc)
    obs = build_observation(features, left_sector=5, right_sector=5)
    history.append(obs)

for step in range(num_steps):
    obs_stacked = history.as_array()
    action, _ = policy.predict(obs_stacked, deterministic=True)
    left_sector, right_sector = int(action[0]), int(action[1])

    data = client.step(left_sector, right_sector)
    features = pre.update(data.left_adc, data.right_adc)
    obs = build_observation(features, left_sector, right_sector)
    history.append(obs)

    logger.write_raw(data)
    logger.write_features(features, action)
```

---

## 22. 附录：推荐的阶段性汇报图

建议未来每次阶段性汇报至少包含：

```text
1. raw ADC 与 baseline 曲线；
2. 预处理后的 signal/smooth/trend 曲线；
3. fixed / periodic / random 的指标柱状图；
4. 不同模式下 sector 时间序列；
5. 左右 smooth diff 曲线；
6. reacquisition time 分布；
7. PPO 与 baseline 的多 seed 箱线图；
8. 触须动作 GIF 或轨迹图。
```

这些图比单纯展示 reward 曲线更能说明主动嗅觉项目的工程价值。
