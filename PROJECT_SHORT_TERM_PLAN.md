# 双触须项目短期工作计划

本文档用于明确当前阶段接下来最实用的短期目标。当前项目已经完成硬件最小链路、触须专用仿真环境、传感器预处理模块和项目路线修正。下一阶段的核心不应是立刻继续扩大 PPO 训练，而是先验证真实 MQ-3 数据经过预处理后，是否能稳定地产生对主动触须采样有用的信息。

## 1. 当前阶段判断

当前硬件链路已经基本跑通：

```text
PC Python
-> STM32
-> 左右舵机
-> 左右 MQ-3
-> 串口返回 ADC
-> PC 保存 CSV / 实时绘图
```

同时，项目已经观察到一个关键现实问题：MQ-3 传感器在洁净空气中也会出现基线漂移，并且每次重新上电后的基线不一致。因此，当前不能把 `left_adc` 和 `right_adc` 原始绝对值直接作为 PPO 状态，也不能直接用它们判断左右哪个方向气体更强。

当前更合理的项目目标是：

```text
验证主动触须采样是否能比固定双传感器提供更高信息密度的局部气味观测。
```

也就是说，短期内不强求触须精确形成“一个指向高浓度、一个指向低浓度”的理想状态，而是先验证主动或扫描触须是否能更容易获得：

```text
气味命中
响应变化
左右不对称
重新捕获气味
值得继续采样的方向
```

## 2. 短期总目标

短期总目标定义为：

```text
用真实硬件数据验证：经过预处理后的 MQ-3 特征是否稳定、可解释，并进一步比较固定传感器、周期扫描、随机扫描三种采样方式的信息获取能力。
```

完成该目标后，再决定是否继续训练和部署主动触须 PPO。

## 3. 阶段一：验证传感器预处理效果

### 3.1 工作目的

这一阶段的目的是确认 `dual_whisker_rl/hardware/sensor_preprocess.py` 是否能把漂移明显的原始 ADC 转换成更稳定、更可解释的特征。

当前传感器预处理模块已经提供：

```text
norm_signal
norm_smooth
norm_trend
norm_estimated_input
左右差分特征
```

但这些参数目前还没有用真实 CSV 系统验证。必须先看预处理结果是否合理，不能直接进入 PPO。

### 3.2 需要完成的工作

新增脚本：

```text
scripts/analyze_hardware_baseline.py
```

脚本输入：

```text
results/hardware/live_plot.csv
```

脚本输出：

```text
results/hardware/baseline_analysis.json
results/hardware/baseline_preprocess.png
```

脚本需要完成以下分析：

```text
1. 读取 CSV 中的 t_pc_s、left_adc、right_adc。
2. 计算左右传感器 raw ADC 的均值、标准差、最小值、最大值。
3. 计算前 10 秒与后 10 秒均值差，估计基线漂移幅度。
4. 计算整体漂移斜率，单位可以是 ADC/s 或 ADC/min。
5. 使用 DualGasPreprocessor 对左右 ADC 做在线预处理。
6. 输出 raw、baseline、signal、smooth、trend、estimated_input 曲线。
7. 输出左右 smooth diff 和 trend diff 曲线。
```

### 3.3 验收标准

这一阶段完成后，需要能回答：

```text
1. 洁净空气中 raw ADC 漂移有多大？
2. baseline 能否跟踪慢漂移？
3. signal 和 smooth 是否比 raw 更稳定？
4. trend 在洁净空气中是否接近 0？
5. 当前 scale、baseline_tau_s、smooth_tau_s 是否需要调整？
```

如果洁净空气中 `norm_smooth` 和 `norm_trend` 仍然剧烈波动，说明预处理参数还不能用于 PPO，需要先调参数。

### 3.4 推荐命令

```powershell
python scripts\analyze_hardware_baseline.py --csv-path results\hardware\live_plot.csv
```

## 4. 阶段二：把预处理接入实时绘图

### 4.1 工作目的

当前 `scripts/hardware_live_plot.py` 主要用于实时查看原始 ADC 曲线。但真实实验中只看 raw ADC 不够，因为 raw ADC 会受到漂移和上电基线差异影响。

这一阶段的目标是让实时绘图同时显示原始读数和预处理后的有效特征，从而在实验现场判断当前气体响应是否真的有信息。

### 4.2 需要完成的工作

修改：

```text
scripts/hardware_live_plot.py
```

接入：

```python
DualGasPreprocessor
```

建议新增命令行参数：

```text
--preprocess
--dt-s
--baseline-tau-s
--smooth-tau-s
--response-tau-s
--trend-window
--scale
```

实时图建议至少包含四个区域：

```text
1. left/right raw ADC
2. left/right baseline 与 signal
3. left/right smooth
4. left-right smooth diff 与 trend diff
```

CSV 中建议额外保存预处理字段：

```text
left_baseline
right_baseline
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
```

### 4.3 验收标准

实时绘图接入后，需要能做到：

```text
1. 实时看到 raw ADC 漂移。
2. 实时看到 baseline 扣除后的相对响应。
3. 实时判断 smooth 是否出现有效气味响应。
4. 实时判断左右差分是否出现方向性信息。
5. 实时保存预处理后的 CSV，方便后续离线分析。
```

### 4.4 推荐命令

```powershell
python scripts\hardware_live_plot.py --port COM3 --preprocess --csv-path results\hardware\live_plot_preprocessed.csv
```

## 5. 阶段三：比较固定传感器、周期扫描、随机扫描

### 5.1 工作目的

这一阶段是当前最关键的验证实验。目标不是训练 PPO，而是先回答：

```text
触须扫描是否真的比固定双传感器获得更多有效气味信息？
```

如果周期扫描都无法比固定角度更好，那么复杂 PPO 主动控制也很可能难以在真实 MQ-3 硬件上稳定体现收益。反过来，如果周期扫描已经比固定角度获得更高命中率、更强趋势或更明显左右不对称，那么说明主动触须控制值得继续投入。

### 5.2 需要完成的工作

新增脚本：

```text
scripts/compare_hardware_sampling_modes.py
```

支持三种模式：

```text
fixed_angle
periodic_scan
random_scan
```

建议命令行参数：

```text
--port COM3
--mode fixed_angle / periodic_scan / random_scan
--duration-s 120
--fixed-sector 5
--cycles 5
--delay-s 0.8
--csv-path
--metrics-path
--preprocess
```

固定角度 baseline 不应只测一个角度。建议至少测：

```text
fixed_30deg
fixed_60deg
fixed_90deg
fixed_120deg
fixed_150deg
```

对应到 10 扇区时，可以先近似选择几个代表性 sector，例如：

```text
sector 1
sector 3
sector 5
sector 7
sector 9
```

实际角度需要与当前扇区中心角映射保持一致。

### 5.3 需要输出的指标

每种模式都输出统一 metrics JSON，建议包含：

```text
sample_count
duration_s
mean_left_smooth
mean_right_smooth
mean_max_smooth
odor_hit_rate
mean_abs_smooth_diff
mean_abs_trend
positive_trend_rate
high_information_rate
reacquisition_events
mean_reacquisition_time_s
```

其中：

```text
odor_hit_rate:
max(left_smooth, right_smooth) > threshold 的比例。

mean_max_smooth:
平均有效气味响应。

mean_abs_smooth_diff:
左右不对称强度。

mean_abs_trend:
响应变化能力。

positive_trend_rate:
采样后出现正向响应趋势的比例。

high_information_rate:
满足高响应、高差分或正 trend 条件的样本比例。

reacquisition_time:
从低响应状态重新超过阈值所需时间。
```

### 5.4 验收标准

这一阶段完成后，需要能形成一张简单对比表：

```text
fixed_angle best
periodic_scan
random_scan
```

如果 `periodic_scan` 在至少部分指标上优于最佳固定角度，例如：

```text
odor_hit_rate 更高
mean_max_smooth 更高
mean_abs_trend 更高
reacquisition_time 更短
```

则说明触须主动或扫描采样有潜力，可以进入下一阶段。

如果 `periodic_scan` 和 `random_scan` 都不如固定角度，则暂时不要训练 PPO，应优先检查：

```text
触须停留时间是否太短
MQ-3 是否响应太慢
传感器位置是否不合理
气源/气流是否不稳定
预处理参数是否过度抑制响应
```

### 5.5 推荐命令

固定角度示例：

```powershell
python scripts\compare_hardware_sampling_modes.py --port COM3 --mode fixed_angle --fixed-sector 5 --duration-s 120 --preprocess
```

周期扫描示例：

```powershell
python scripts\compare_hardware_sampling_modes.py --port COM3 --mode periodic_scan --cycles 5 --preprocess
```

随机扫描示例：

```powershell
python scripts\compare_hardware_sampling_modes.py --port COM3 --mode random_scan --duration-s 120 --preprocess
```

## 6. 阶段四：调整仿真 PPO 的观测形式

### 6.1 工作目的

只有当硬件预处理和采样对比实验初步证明“触须扫描确实能产生有效信息”后，才建议继续调整和训练触须 PPO。

这一阶段目标是让仿真 PPO 的观测更接近硬件部署时的观测。

### 6.2 建议观测

后续触须 PPO 观测应尽量使用硬件可获得的特征：

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

不要让策略过度依赖仿真中特有的信息，例如：

```text
真实气源距离
真实风向
真实瞬时浓度 raw concentration
```

这些信息在真实硬件上难以直接获得，如果训练时依赖它们，部署时会出现观测分布不一致。

### 6.3 历史信息

继续使用简单历史拼接，不急着使用 LSTM/GRU：

```text
history_length = 1
history_length = 10
history_length = 20
history_length = 30
```

推荐优先尝试：

```text
history_length = 20
```

因为当前触须仿真 `dt = 0.2s`，20 帧约等于 4 秒历史，比较适合 MQ-3 的慢响应特性。

### 6.4 验收标准

重新训练 PPO 之前，需要确认：

```text
1. 仿真观测维度明确。
2. 训练时和硬件部署时可以构造同样含义的状态。
3. run_metadata.json 能记录 history_length 和观测字段。
4. 可视化图能显示 PPO 选择扇区、传感器 smooth/trend 和左右差分。
```

## 7. 阶段五：再考虑主动 PPO 与硬件策略推理

### 7.1 进入条件

只有满足以下条件后，才建议进入主动 PPO 硬件部署：

```text
1. 预处理后的硬件特征稳定可解释。
2. 周期扫描或随机扫描能在部分指标上优于固定传感器。
3. 仿真 PPO 的观测形式已经接近硬件可部署状态。
4. PPO 在仿真中相比固定/随机策略有可观察提升。
```

### 7.2 后续脚本

再新增：

```text
scripts/hardware_whisker_policy_rollout.py
```

基本流程：

```text
1. 加载 PPO 模型。
2. 打开串口。
3. 读取 STM32 返回的 ADC。
4. 使用 DualGasPreprocessor 做预处理。
5. 维护 history queue。
6. 拼接 PPO 输入状态。
7. PPO 输出 left_sector/right_sector。
8. 串口发送 STEP left right。
9. 保存 CSV、metrics 和实时图。
```

## 8. 推荐执行顺序

建议严格按下面顺序推进：

```text
1. scripts/analyze_hardware_baseline.py
2. scripts/hardware_live_plot.py 接入预处理
3. scripts/compare_hardware_sampling_modes.py
4. 固定角度 vs 周期扫描 vs 随机扫描硬件实验
5. 根据实验结果决定是否调整硬件采样节奏和预处理参数
6. 调整 WhiskerOnlyPuffEnv 的 PPO 观测形式
7. 重新训练 whisker-only PPO
8. 编写 hardware_whisker_policy_rollout.py
```

当前最优先的一个任务是：

```text
实现 scripts/analyze_hardware_baseline.py，并用 results/hardware/live_plot.csv 验证传感器预处理模块。
```

这是后续所有主动触须训练和硬件部署的地基。先把真实传感器数据看明白，后面的 PPO 才不会变成盲目调参。
