# 延期的真实硬件实验

当前只开展仿真对比，本文件保留E0与E7的后续设计，不属于当前执行流程。硬件采集与分析脚本保留供后续阶段使用，当前仿真入口不再提供hardware-plan阶段。

## 4 静态标定与漂移

### 4.1 生成排程

硬件排程设计暂存于dual_whisker_rl/experiment_protocol.py的hardware_schedule函数；恢复硬件阶段时再接入专用入口。

生成results/red_experiments/formal/hardware下的三个CSV：
- static_schedule.csv：3天×3循环×升/降序×6水平，共108个平台，其中90个非零平台、18个空白平台。
- drift_schedule.csv：3天各冷启动、稳定空白2小时、稳定中浓度2小时，共9条记录。
- real_schedule.csv：4方法×3条件×30次，共360次，按条件块随机运行顺序。

排程中的day_1等是安排标签，应在实采时填写真实ISO日期。reference_value留空，必须由参考仪器获得；target_reference仅表示设定值。重新生成排程不会覆盖已有CSV，已填写的记录仍建议另存real_results.csv。

### 4.2 先检查，再采一段

~~~powershell
.\.venv\Scripts\python.exe scripts\collect_sensor_experiment.py --run-id d1_up1_100 --dry-run

# 按排程，在供气稳定后采集一个平台；该命令会真实打开串口并驱动触须
.\.venv\Scripts\python.exe scripts\collect_sensor_experiment.py --port COM3 --run-id d1_up1_100 --mode fixed --duration-s 120 --period-s 0.8 --date 2026-10-13 --plateau-id d1_up1_100 --sequence-id d1_up1 --direction ascending --target-reference 100 --reference-unit ppm

# 稳定空白2小时；冷启动记录与稳定记录分开
.\.venv\Scripts\python.exe scripts\collect_sensor_experiment.py --port COM3 --run-id d1_warm_blank --mode fixed --duration-s 7200 --period-s 0.8 --condition warm_blank
~~~

数据在results/sensor_experiments/<run-id>/raw.csv，旁边保存metadata.json和state.json。CSV含左右原始ADC、命令/响应单调时钟、串口RTT、真实采样周期、超时周期标志、平台/序列/日期和目标扇区。温湿度、参考值与电压未知时保持空值，不能补成零。

当前STEP命令需等待舵机稳定后返回。--period-s是目标周期；脚本不会宣称实际达到50Hz。真实周期来自返回时间差，串口耗时超过目标则记录period_overrun。

### 4.3 合并外部参考并标定

把所有原始平台CSV保留不动。另建reference.csv：

~~~csv
plateau_id,reference_value,reference_unit
d1_up1_100,98.7,ppm
~~~

参考表应覆盖全部平台、ID唯一，并来自同步参考测量。如果只有可重复释放档位，reference_unit使用release_level，结果不能写成ppm。

~~~powershell
.\.venv\Scripts\python.exe scripts\analyze_sensor_experiments.py static --input results/sensor_experiments/d1_up1_100/raw.csv results/sensor_experiments/d2_up1_100/raw.csv results/sensor_experiments/d3_up1_100/raw.csv --reference-csv results/sensor_experiments/reference.csv --holdout-date 2026-10-15 --output-dir results/sensor_analysis/static_01
~~~

上述输入路径仅演示多文件用法：正式分析必须包含训练日至少3个不同参考水平、完整升降序和留出日。只有三条相同浓度记录会被拒绝。

末段60秒是分析窗口，不是自动证明MQ-3已经稳态；先按冻结变化率和波动门限确认稳态。程序按平台末段60秒汇总，平台内参考变化超过5%则拒绝稳态标定。拟合ADC到参考值的1/2次曲线，非单调二次曲线回退到一次。整天留出预测，输出双通道系数、适用ADC范围、MAE/RMSE、滞回和平台计数；系数只使用训练日拟合，不外推。不把同一平台多个采样点算作独立标定重复。

~~~powershell
.\.venv\Scripts\python.exe scripts\analyze_sensor_experiments.py drift --input results/sensor_experiments/d1_warm_blank/raw.csv --scale 200 --block-s 60 --output-dir results/sensor_analysis/drift_01
~~~

漂移分析输出ADC/hour、工作信号尺度百分比、残差时间块bootstrap区间及frozen/gated/relaxed三种基线处理。gated对应项目现有门控预处理，relaxed放宽门限，frozen不更新。基线锚定各记录第一样本；比较持续气味的保留幅度时，必须用包含空白锚定阶段的连续记录，不能从已经处于气味中的第一点判断目标幅度。输出cv是原始ADC的CV，不能替代非零响应幅度的重复性CV。原始斜率仍可能包含参考浓度、温湿度或电源变化，不自动等同于器件漂移。

## 8 E7硬件采样与移动找源边界

现有固件只有STEP left right和双ADC返回，没有底盘运动命令、航向反馈、源真值定位和安全停止接口。因此：

1. fixed/periodic/random采样可直接使用collect脚本，比较同日、同风源、同位置下的ADC和气味信息。
2. PPO采集只接受动作MultiDiscrete([10,10])、单帧12维观测的固定基座模型；拒绝16维/三动作头的Mobile模型。
3. 移动实机E7目前提供360次排程与结果导入分析，不声称已经实现底盘闭环搜索。接入底盘和独立真值系统后再填写真实结果。
4. 返回的sector是指令回显，不是舵机角度反馈。日志明确标记command_estimate，实机角度误差需独立测量。

PPO预处理YAML需要SensorPreprocessConfig参数，另加实测left_baseline、right_baseline。未标定时不要照抄scale=200。PPO命令必须声明training-period-s且匹配采样周期；实测周期偏离超过15%会停止并保留记录。H20在0.8秒硬件周期下是约16秒，不是4秒；当前固件不能通过脚本复制ADC伪造0.2秒新观测。

~~~powershell
# 只在已有匹配时钟、冻结预处理的固定基座模型后执行
.\.venv\Scripts\python.exe scripts\collect_sensor_experiment.py --port COM3 --run-id ppo_R1_001 --mode ppo --model results/models/whisker_only_ppo.zip --preprocess-config configs/hardware_preprocess_measured.yaml --period-s 0.8 --training-period-s 0.8 --duration-s 120
~~~

上述模型与预处理文件必须来自实际实验，示例文件不会自动生成。模型若训练于0.2秒观测间隔，不得填0.8秒使检查通过，应先按硬件节奏重新训练。

真实移动实验完成后，另存real_results.csv，填写success、termination、elapsed_s及真值指标。失败、人工干预和安全停止全部保留。

~~~powershell
.\.venv\Scripts\python.exe scripts\analyze_sensor_experiments.py trials --input results/red_experiments/formal/hardware/real_results.csv --horizon-s 80 --output-dir results/red_analysis/real_01
~~~

学习方法按训练seed、匹配条件块配对；规则法没有训练seed，不制造5个规则模型。当前通用表对规则法不能构成种子配对时保持unpaired，不能据此声称真实四组已完成混合效应推断。需要按日期/条件块的真实主统计模型时，以全部原始试验表另行拟合并预注册。

