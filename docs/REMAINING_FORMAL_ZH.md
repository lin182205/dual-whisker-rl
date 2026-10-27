# 当前执行协议

旧训练封存继续使用，新正式评估已切换为修正羽流与随机采样；现行命令与接管规则见[新协议指南](STOCHASTIC_REEVALUATION_ZH.md)。下文保留原流水线设计记录。

# 先训练单帧MLP，再补齐剩余正式实验

此入口执行已经确认的服务器计划：新训练一个1500万步、单帧H=1、16维观测、无DR的MLP/PPO模型，
随后补充主对比与E3报告，完成原九方法E4、完整E5消融、核心方法E6和两种长训练模型的补充扰动。
所有工作仅为仿真，E0/E7继续延期。

## 启动与恢复

在服务器项目目录、已有Conda环境中执行：

```bash
# 只读检查来源、预算及剩余训练数量
bash scripts/run_remaining_formal_conda.sh plan --sealed-source results/remaining_formal_sources_20261027
# 后台执行；断开SSH后继续，同一命令也用于明确恢复
bash scripts/run_remaining_formal_conda.sh run --background --sealed-source results/remaining_formal_sources_20261027
# 读取流水线状态
bash scripts/run_remaining_formal_conda.sh status --sealed-source results/remaining_formal_sources_20261027
```

Python入口支持plan/run/status，默认plan；启动器默认run并自动携带--resume。
只有run支持--background。提交后台PID不表示训练已启动，须核对主日志和state.json。
默认输出为results/remaining_formal_15m；日志入口保存在results/logs/remaining_formal/latest.log。
其中index.md链接各实验报告，state.json记录每个阶段，processes保存子进程与退出码。

新模型使用single_frame_mlp_15m关键词分配新编号；模型、checkpoint、TensorBoard和元数据
保存到现有results/models与results/logs体系。mlp_training.json在开始训练前固定真实seed、
实验编号和目标总步数。中断后恢复只计算1500万减checkpoint实际累计步数，不额外再加1500万。
PPO按完整轮次取整，默认12×512步时最终预计15,003,648步，必须以实际模型头为准。

## 封存隔离与预算

原生formal使用第2版封存，新调度不修改旧manifest、不把它升级为第3版。
--sealed-source必须提供与旧manifest逐字节一致的科学源码、原运行器，以及原E4入口、
experiment_progress.py和e4_formal.yaml。部署包携带这些源码；来源不全或SHA-256不符就退出。
运行时创建sealed_v2执行目录，仅含原封存的科学模块；辅助调度文件不进入原科学签名。
旧逐局与模型保持原位，只追加缺少的配置或场景；报告另存，不覆盖旧summaries或75M报告。

初次核验30个核心训练已完成时，剩余为27配置×5seed×200万步、原E4的35次100万步和校准，
再加新MLP1500万步，合计约3.2亿环境步。每次plan重新核验完成训练，不把这当作固定待运行量。
原E4保持完整九方法与原方法种子安排；规则方法不算独立训练模型。

E5覆盖历史、特征、奖励、几何、传感器、域随机化六类。E6采用nominal、wind24、gain80、
servo150、delay2五条件，每个模型/条件取同一测试库前200个场景。核心组为6方法×5seed，
总计30,000局。75M GRU与新15M MLP各自另做5×200局，使用首次到源判据并单列报告。

新MLP在648个原共享场景评估，七组主对比复用既有16,848局，仅新增648局，共17,496局。
训练预算与成功判据不同，七组报告描述冻结模型集合，不声称匹配预算的算法因果效应。
长训练模型每个只有一次训练；区间条件于该模型，不伪造5个训练seed。

## 监控与验收

一次只调度一个长训练子进程，文件锁阻止重复流水线；已知监控子进程仍存活时拒绝重新调度。
每60秒检查进程和声明的日志，60分钟无输出增长只告警。没有自动终止、自动重试或自动删锁。
主监控器中断后，子进程可能仍活着；先检查processes记录，不要直接重新启动另一条训练。
失败阶段写入退出码与日志路径，后续阶段不再执行；修复后显式恢复。

训练完成要求最终模型、实际步数、真实seed、16维单帧MLP接口与元数据一致。
原生E5/E6核对逐场景签名、模型摘要和压缩轨迹行数；原E4沿用逐局JSON，
核对验证/测试场景摘要、逐局签名和训练预算，不声称原E4已保存逐步压缩轨迹。
报告覆盖率和必要对照不全时不会标记完成；缺项保留为空，不计作失败或填零。

--stages可指定阶段进行诊断；部分阶段完成只标selected_stages_completed。
--mlp-timesteps、--mlp-n-envs、--mlp-n-steps与--mlp-checkpoint-freq用于明确预算或软件短验证，
改变这些参数须使用新输出目录。--workers仅控制补充评估，原正式训练仍固定原8环境协议。
人工QA结果与正式目录分开保存，不能把短预算测试图用作论文效果数据。
