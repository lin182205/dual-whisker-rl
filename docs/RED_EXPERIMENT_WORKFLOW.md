# 仿真对比实验流程

当前75M主方法主对比请使用[快速对比指南](FAST_COMPARISON_ZH.md)：进入0.1米半径即成功，转换旧轨迹并只补缺项，不启动完整E3–E6训练。下文完整实验协议继续保留，历史结果不覆盖。

最新命令与图表说明见[中文使用指南](SIMULATION_EXPERIMENT_GUIDE_ZH.md)。现在省略--experiments时默认运行E3–E6全部；可用--experiments e3 e6选择。每次执行自动创建独立的reports/<UTC时间戳>图表目录。

当前阶段只开展E3训练稳定性、E4仿真主对比、E5消融与敏感性、E6域外扰动仿真。研究重点是主动触须在间歇羽流和慢响应传感器模型下的信息获取收益，移动找源指标作为任务终点同时报告。

配置scope固定为simulation_only。全部命令通过Gymnasium环境运行，无需真实传感器、舵机、底盘或串口。传感器模型采用现有标称参数0.6/3.0秒及其敏感性范围，这些数值属于仿真假设，当前阶段不要求完成实测标定，也不据此宣称实机迁移有效。

所有结果默认保存在results/simulation_experiments；以前的结果继续保留在各自目录。本次执行范围和入口版本进入封存签名，新流程使用新的默认目录。原E4仍采用其既有运行器和判据。

## 1 文件与实验对应

| 文件 | 用途 |
|---|---|
| configs/red_experiments.yaml | smoke/formal预算、35个策略配置与故障水平 |
| scripts/run_red_experiments.py | 仿真计划、封存、训练、评估、扰动仿真、汇总及E4委托 |
| scripts/experiment_progress.py | 评估进度、耗时与剩余时间显示，区分新场景和复用结果 |
| scripts/analyze_experiment_learning.py | 跨训练seed学习曲线、AUC、塌缩诊断与故障退化 |
| dual_whisker_rl/experiment_env.py | 特征屏蔽、几何、域随机化幅度、持续到源、传感器/风场/命令故障 |
| dual_whisker_rl/experiment_protocol.py | 封存、交叉bootstrap、种子级检验与Holm校正 |
| tests/test_red_experiments.py | 环境口径、名义等价、故障复现和持续到源检查 |

## 2 首次运行

在仓库根目录使用项目虚拟环境。现有requirements已覆盖新增脚本，未添加新的依赖包。

~~~powershell
# 查看计划；不训练、不生成模型
.\.venv\Scripts\python.exe scripts\run_red_experiments.py plan --profile smoke

# 固定验证/测试场景与代码、配置摘要
.\.venv\Scripts\python.exe scripts\run_red_experiments.py prepare --profile smoke

# 跑全部E3–E6的短训练、评估与图表报告
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --profile smoke --resume

# 查看任务状态
.\.venv\Scripts\python.exe scripts\run_red_experiments.py status --profile smoke
~~~

smoke默认：1个训练seed、每配置256步、1个环境、4个验证场景、8个测试场景、每场景最多40步。它只验证连通性。不能用smoke的成功率或CI作为论文结论。

formal默认：5个独立训练seed、每配置200万环境步、8个spawn worker、固定200个验证场景、648个共享测试场景、每局400步。训练在支持的设备上使用auto；smoke使用CPU。以总环境交互计数，不能把8个worker分别计作8份预算。

首次正式执行前，确认当前羽流覆盖和间歇结构、仿真读数分布、奖励量级与统一训练预算。YAML中的传感器时间常数和触须速度是当前标称建模值，通过E5/E6扫参检验结论对假设的依赖。

### 2.1 评估进度与预计剩余时间

无需额外参数，evaluate、robustness以及run中的评估阶段自动打印进度。例：

~~~text
[评估] 组=2/30 fixed seed=11 fault=nominal 场景=130/19440 (0.7%) 本次新增=30 复用=100 已用=00:02:10 预计剩余=23:14:09
~~~

这是一条格式示例，不是实测实验结果。组表示方法×训练seed×故障组合，场景进度和预计剩余覆盖本次命令选中的全部组合。首个新场景完成前显示“等待首个新场景完成”；完成后，按本次新场景的平均实际耗时乘待运行场景数估时，包括推理、仿真和逐场景写盘。首个新场景、最后一个场景立即刷新，其余在场景完成时检查，每新增20场景或距上次输出满10秒刷新。时长格式为小时:分钟:秒，小时可以大于24。

--resume先校验并计数已有结果；复用数进入总进度，读取旧结果的耗时不作为新评估速度。全部复用时显示100%和剩余00:00:00，并跳过模型加载和仿真运行。估时会随提前结束、不同策略推理耗时和机器负载变化；模型加载与后续汇总开销不在场景平均耗时内。run的提示只覆盖评估阶段，训练预算不计入这一倒计时。

原scripts/run_e4_experiments.py按当前方法×seed×数据集打印“本组”剩余时间，每新增50场景或满10秒在场景完成时刷新；其缓存速度计算已修正。scripts/evaluate_whisker_sampling_policies.py按选中策略×回合数打印本轮总进度，使用20场景/10秒规则。重定向日志使用UTF-8，交互式Windows终端沿用本地编码。

本次已知、仅增加显示的运行器升级支持旧仿真封存：逐字段校验全部配置、代码、场景和旧签名后保留manifest与模型原签名，另写reporting_upgrade.json记录当前运行器摘要。该例外只绑定升级前后两个精确运行器SHA-256；任何其他代码、配置、场景或预算变更仍拒绝混用旧结果。

### 2.2 直接复评已有训练模型

--model-path接受已有SB3 PPO模型，可用绝对路径或相对仓库根目录的路径；.zip后缀可省略。
run传入它后跳过训练，按--experiments选择复评与报告；普通复评用--experiments e4，扰动用--experiments e6。缺少日志或重训对照时，E3/E5输出缺项表。robustness仅执行E6。
未指定--variants时只评估main；其他方法须显式指定单个variant，例如history_mlp。一个路径不能
用于整组或多个variant。train和原E4委托阶段不接受该参数。

~~~powershell
# 先检查计划：training_tasks为0，训练seed从模型读取
.\.venv\Scripts\python.exe scripts\run_red_experiments.py plan --profile formal --model-path results/models/size_5-gru_3.zip

# 快速验证已有模型的评估链路，不重新训练
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 --profile smoke --model-path results/models/size_5-gru_3.zip --resume

# 648个正式共享测试场景，只评估这个模型
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 --profile formal --model-path results/models/size_5-gru_3.zip --resume

# 同一模型、同一批扰动测试场景；默认每故障200场景
.\.venv\Scripts\python.exe scripts\run_red_experiments.py robustness --profile formal --model-path results/models/size_5-gru_3.zip --faults nominal,wind24,gain80,servo150,delay2 --resume
~~~

默认输出为results/simulation_experiments/pretrained/<模型名>/<profile>；--output-dir可指定
独立目录，仍自动附加profile。即使formal配置有5个seed，已有模型也只按文件保存的真实训练
seed执行一次，不把同一模型伪装成5次独立训练。未保存seed时须以--model-seed声明；已保存
的seed不能改写。模型原文件只读，不复制或生成selected/final/last模型，不创建训练完成状态
或学习曲线。status在该模式下显示来源信息。

manifest封存模型路径、SHA-256、训练seed、实际训练步数和当前评估配置；逐场景JSON和汇总
同样保留来源。--resume只有在模型、代码、配置和场景均匹配时复用，全缓存不加载策略。
改换模型、改动文件内容或评估配置后用新输出目录。此次新增模型导入功能不属于§2.1的显示
兼容例外，不会放宽旧实验封存；旧目录中的selected_model.zip可通过--model-path在新目录复评。

观测/动作空间、GRU历史长度/单帧维度和编码器均需匹配所选variant；旧15维Mobile模型不能
直接用于当前16维观测。接口匹配只证明能执行，training_config_verified始终记录false：用户仍需
核对原训练场地、羽流、观测语义和传感器参数。新入口仍采用持续1秒成功判据，旧模型复评
不是统一200万步重训或验证集选择结果，也不补出训练稳定性或消融证据。

## 3 实验定义与复现规则

新增实验采用0.2秒物理/观测周期，H20名义4秒历史。成功要求进入goal_radius并连续保持1秒，奖励到源项只在合格后发放。搜索时间为实际步数乘dt；失败惩罚时间在任何失败时均取max_steps乘dt，不让快速越界得到较短搜索时间。

原E4的成功判据是瞬时进入半径。两种判据的结果必须分开报告；E4任务不可与新增红字入口的成功率直接混合汇总。项目既有严格重捕获门限1秒保持原口径；新脚本不会暗改成方案中候选的2秒。调整门限需在正式封存前统一修改并说明。

manifest.json封存完整配置、物理参数、全部策略与故障定义、源代码SHA-256、验证/测试场景。所有策略使用相同测试场景，羽流随机流与策略/传感器随机流分离。评估时关闭额外训练域随机化；DR表示训练方法的不同，不表示测试时给某组额外随机题。

- 修改源代码、配置或预算后使用新output-dir；签名不同会拒绝恢复。
- --resume保留完成任务，未完成训练使用last_model.zip继续；不自动重试失败。
- 训练checkpoint在已完成PPO更新的轮次边界保存。重启不保存VecEnv内部状态，续训不保证逐位一致。
- 以固定验证成功率、失败惩罚时间、较早checkpoint为规则选selected_model.zip，并保留final_model.zip。
- 测试集和域外扰动集不能用于选择checkpoint、固定角度或扫描周期。
- 同目录的train/evaluate/run/robustness使用运行锁。进程正常或异常退出时清理；操作系统强杀留下锁时，先确认原进程结束，再删除该目录的.run.lock。
- 逐场景JSON记录模型摘要、故障和评估签名；--resume不会静默接受不同模型的旧评估。
- 分析输出使用新目录，保留原始逐场景数据。

## 4 E3与E5训练和消融

~~~powershell
# 主方法与核心对照
.\.venv\Scripts\python.exe scripts\run_red_experiments.py train --experiments e3 --profile formal --group core --resume

# 各因素逐组执行，共享封存场景
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group history --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group features --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group rewards --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group geometry --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group sensor --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --group randomization --resume
~~~

core：main、fixed、periodic、single_frame、history_mlp、no_dr。history包含H1/5/10/20/40；features包含趋势、显式差分、blank_age、角度屏蔽与原始读数；rewards包含去重捕获、最佳浓度、停滞和稀疏奖励；geometry包含四种长度与3/5/10/15扇区；sensor包含GRU/MLP×理想/标称动态的2×2（ideal_sensor、ideal_mlp、nominal_gru、nominal_mlp）以及tau扫参；randomization包含无、窄、默认、宽。

固定角度和扫描周期在YAML中显式设置。默认fixed_sector=5、scan_hold=2仅是候选，不是验证得到的最优值。正式前用验证实验校准，或使用既有E4的校准结果；填入配置后再封存。不能把未调参的固定组称“最强基线”。MLP的参数匹配是启发式，metadata保存实际参数量；不是保证±10%。

每次验证写validation_history.json，包含成功率、信息获取、失败惩罚时间、奖励分量、PPO loss/KL/clip等、梯度范数及动作头熵。selected_model、final_model、last_model各自分开。对照重新训练，不能只在完整模型推理时删特征。

~~~powershell
.\.venv\Scripts\python.exe scripts\analyze_experiment_learning.py learning --experiment-dir results/simulation_experiments/formal --output-dir results/simulation_analysis/learning_01
~~~

学习曲线CI按训练seed重采样；单seed不输出算法级CI。AUC只覆盖已有验证点区间，不虚构0步表现。达标使用连续三次验证≥80%作为候选指标；输出塌缩和后期波动。所有阈值是实验定义，不是投稿保证。

## 5 E4复用原运行器

~~~powershell
.\.venv\Scripts\python.exe scripts\run_red_experiments.py e4 --profile smoke --e4-stage prepare
.\.venv\Scripts\python.exe scripts\run_red_experiments.py e4 --profile formal --e4-stage calibrate --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py e4 --profile formal --e4-stage run --resume
~~~

委托scripts/run_e4_experiments.py及当前configs/e4_*.yaml，输出独立放在红字profile目录的e4子目录。原E4训练预算、基线校准、恢复、规则法和统计实现保持原有口径，不能将它的结果误称为新增200万步训练或持续1秒成功实验。

## 6 E6冻结模型后的扰动仿真

先完成选定策略的训练。nominal必须与各故障使用同一批共享场景，才能计算退化。

~~~powershell
.\.venv\Scripts\python.exe scripts\run_red_experiments.py robustness --profile smoke --variants main,no_dr --faults nominal,wind24,gain80,servo150,delay2 --limit 3 --resume
.\.venv\Scripts\python.exe scripts\run_red_experiments.py robustness --profile formal --variants main,no_dr,fixed,single_frame --faults nominal,wind08,wind24,wind28,turn15,turn30,turn45,source25,gain70,gain130,tau200,servo150,delay4,packets10 --resume
.\.venv\Scripts\python.exe scripts\analyze_experiment_learning.py robustness --experiment-dir results/simulation_experiments/formal --variants main,no_dr,fixed,single_frame --output-dir results/simulation_analysis/robustness_01
~~~

本节全部为软件扰动仿真：sensor_tau模拟慢响应，servo_time_factor模拟动作限速，packets模拟命令丢弃，delay模拟命令延迟队列，不涉及真实设备或串口。formal默认选200个共享测试场景，每方法每故障5seed×200次。故障不会重新训练或选择模型。支持风速、风向跳变、源强/间歇释放、读数增益/漂移/饱和/丢样、传感器tau/噪声、舵机变慢、命令延迟/丢包与在线航向噪声。tau、噪声和舵机参数失配从reset开始；start_s用于读数变换、命令故障、风向改变与释放门控，不能把这些静态参数变化解释成统一时刻发生。

丢样保持上一读数，物理传感器继续响应；没有NaN进入策略。命令延迟以物理tick实现，不以真实time.sleep模拟。故障独立随机流保证同场景复现。位置真值不加噪来制造“策略不鲁棒”。

报告绝对值、百分点差和对nominal的比值；时间比值越大越差。基线接近零时比值为空。一般故障扰动不一定都是严格OOD，必须对照实际训练DR支持范围；尤其源强与噪声部分水平可能位于训练域内。不要根据故障测试重新调DR范围后仍称盲测试。

## 7 统计与结果检查

- 先看odor_hit_ratio、触须独立重捕获、正响应趋势和左右信号差，再结合移动任务成功率、误差和失败时间。
- summaries中的coverage明确期望/实际局数；未完成的组不能包装为完整实验。
- 消融差值按共享场景和训练seed交叉bootstrap；种子级双侧符号置换检验在5seed时最小p为0.0625。
- Holm在每终点内的预定对比族校正；跨多个主要终点的正式检验族还需实验前指定。
- 没有重捕获事件的时间保持缺失，不填零。
- 不同奖励组比较任务与信息指标，不直接比较奖励数值。
- summaries的nominal_episodes.csv保留逐局数据，JSON保留差值和CI；steps.jsonl.gz保留观测、动作、传感器、轨迹、风和奖励分量。
- 日志大时规划磁盘空间。正式save_step_logs=true会保存全部仿真轨迹，不能只留下图表均值。
- 实验输入与模型摘要写入manifest.json和逐场景记录，学习曲线与退化分析保留对应场景和种子。

## 8 验证入口

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_red_experiments.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment_progress.py -v
.\.venv\Scripts\python.exe scripts/check_portable_paths.py
~~~

代码和短训练通过只说明仿真实现与记录链路可用；科学结论需要正式多seed、共享场景和完整统计。E0真实标定/漂移与E7实机验证延期，既有设计存放在[延期硬件实验](DEFERRED_HARDWARE_EXPERIMENTS.md)，当前流程不执行这些步骤。
