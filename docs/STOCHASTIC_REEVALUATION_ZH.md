# 修正羽流与随机动作的正式重评

## 本轮协议

统一首次进入气源0.1米即成功、无需保持，评估不再叠加训练域随机化。
PPO从其Categorical/MultiCategorical策略概率采样动作，不增加温度或额外探索噪声。
规则方法保持原规则；随机规则的各个种子为不同控制器重复，不是独立训练。

训练环境会根据世界范围与最低风速派生气团寿命，5×5米、最低0.12米/秒时为45.8333秒。
旧实验适配器重建羽流时遗漏派生参数，使用了7秒。本轮改为首次构造即选择实验羽流，
继承完整解析参数与域随机化基准；显式覆盖参数仍优先。旧10.2%来自7秒气团、确定性
动作，保留为历史条件结果，不与本轮合并。

## 已批准的执行顺序与命令

先完成正在运行的15M单帧训练，再重评七组主对比，随后继续原E4/E5/E6训练计划与新评估。
75M与历史模型不重新训练。训练保留原封存、目标总步数和checkpoint选择规则，训练内的
历史验证不改写成新正式评估。旧2M模型的训练物理与75M不同，报告须继续披露。

2026-10-28起，E4暂定每种需训练方法、每个seed **8,000,000步**，7种方法×5seed，
主体预算280,000,000步；随机/反应式规则不训练。固定触须扇区校准仍为原短预算。
8环境×512步rollout取整时，每模型预计8,003,584步，以实际模型头记录为准。
原100万步E4准备目录只读保留，新预算训练写入results/e4_8m/formal，不改写旧封存。
预算修改不改变现有模型选择规则、PPO参数、15M训练或E5预算。

在部署好的服务器发布目录中运行：

```bash
bash scripts/run_remaining_formal_conda.sh run --background \
  --evaluation-protocol stochastic-arrival-v2 \
  --sealed-source results/remaining_formal_sources_20261027 \
  --adopt-training-run results/remaining_formal_15m \
  --output-dir results/reevaluation_stochastic_v2_e4_8m_parallel_recovery \
  --stages train_mlp,e4,e5,e6,long_faults \
  --e4-timesteps 8000000 --e4-training-dir results/e4_8m/formal \
  --e4-concurrency auto --e4-probe-seconds 150 \
  --episode-store results/reevaluation_stochastic_v2/episodes \
  --workers auto --device auto --eval-seed 20261027 --eval-repeats 1
```

将`run --background`替换为`plan`或`status`可只读检查。新Python入口默认新协议，
默认只计划；后台启动器默认执行。不要同时启动第二个父调度器。
上面的阶段列表用于当前服务器恢复：主对比和E3已经完成，当前目录保留其报告引用。
在全新实验上执行完整流程时应省略--stages，并选择全新输出目录。

仅重评75M与五类对照、不涉及训练时，使用：

```bash
bash scripts/run_fast_comparison_conda.sh run --background \
  --evaluation-protocol stochastic-arrival-v2 \
  --sealed-source results/remaining_formal_sources_20261027 \
  --output-dir results/reevaluation_stochastic_v2/main_six_only \
  --episode-store results/reevaluation_stochastic_v2/episodes
```

完整流水线会加入15M模型；单独对比入口的六组封存与七组封存不同，不能在已经封存的
七组目录内启动六组入口。六组入口单独使用时请另选输出目录。

## 接管、恢复与缓存

部署时核对PID、进程创建时间与命令，仅暂停旧父调度器，训练子进程继续运行。
新调度器读取handoff.json等待训练结束，验证最终模型、实际步数、seed和元数据后退出
旧父调度器并接管。若训练失败则停止后续阶段，不擅自从checkpoint恢复或增加预算。
原工作目录在当前训练完成后才同步；部署前差异保留备份及Git stash。

当前新输出根目录为`results/reevaluation_stochastic_v2_e4_8m_parallel_recovery`；
实际活动目录以`results/active_formal_campaign.json`为准，先前调度封存保留。
通过--episode-store继续共享原`results/reevaluation_stochastic_v2/episodes`，避免预算
调度更新导致相同科学协议的主对比/E5/E6重复评估。episodes保存共享逐局JSON与
压缩轨迹；main、e4、e5、e6、long_faults保存各自视图与报告；index.md提供报告索引。
旧manifest、旧轨迹、旧报告均保留，不将旧轨迹截断或重标为新物理条件下的轨迹。

默认评估种子20261027，每模型/场景采样一次。种子由评估种子、模型摘要、场景标识/种子
及重复编号稳定派生，故障不参与动作随机流的派生，以便同模型故障配对。
各随机规则重复的控制器身份包含其原种子。更换并行度或调度顺序不改变逐局随机数。

缓存键包含模型、有效环境、场景、故障、采样配置与科学实现。新协议完全一致才可复用。
每阶段task_plan.json列出实际报告任务数、独立任务键、缓存数与新增评估数；完整缓存
不加载策略。缺失或损坏逐局文件只补对应任务。模型/题库/实现变动时使用新封存目录，
不能放宽旧签名。中断恢复前检查processes，存活子进程会阻止重复调度。

## 资源与报告解释

E4并行覆盖B1规则搜索、B2的20个102,400步短校准任务、7方法×5seed的35个正式模型。
`--e4-concurrency auto|1|2`默认auto；auto在独立诊断目录测量单GRU、单历史MLP、双GRU、
双MLP及混合组合。`--e4-probe-seconds 150`平均分配为每配置30秒；初始化和一个完整
rollout预热不计时，计时包括后续采样和PPO更新，并允许完整rollout收尾。因此总墙钟
时间会超过150秒。只看吞吐，不使用正式测试场景或成功率选择配置。

同类双模型合计吞吐至少超过对应单模型10%，混合组合至少超过两个单模型吞吐的
调和平均10%，才允许该组合并行。最大两模型，各自仍用8环境、512步及原PPO参数，
Torch和数学库线程数均为1。80%容器内存上限及至少1GiB显存余量约束第二任务启动。
资源不足则等待；吞吐不达标的组合串行。B1按CPU配额、内存及--workers限制分配候选。
显式1/2可覆盖吞吐选择，资源余量约束仍然有效。正式评估与训练分阶段执行。

E4调度记录在`results/e4_8m/formal/parallel/`：`concurrency_probe.json`保存硬件、软件、
计时、实际步数及允许组合；`processes/`保存每任务PID、创建时间、命令及状态，
`logs/`保存独立日志，`queue_state.json`保存当前队列。正式模型、TensorBoard和
checkpoint仍在原方法/seed目录。诊断模型不计入正式35个模型。校准按原候选、扇区和
seed顺序汇总，完成顺序不影响均值、优先级或平局规则；两项校准完成后才能正式训练。

E4根目录使用独占调度锁；恢复先识别存活任务，避免重复训练。硬件或训练/诊断实现改变
需要重测时，先等存活任务完成再测。失败会停止派发，继续监控健康任务，不自动重试；
人工查明失败原因并保存原收据后才可恢复失败任务。已有checkpoint恢复至800万总目标，
不是再追加800万。切换父调度保留B1结果、已完成校准模型及验证缓存；新调度仅补缺项。
Linux新增进程启动tick与内核启动编号作为稳定身份；保留墙钟创建时间供审计。
自建子进程使用操作系统poll确认退出，身份暂时不可读时继续占用槽位，不等待5秒后退出调度。

长训练与正式评估串行。评估按亲和性、配额、内存和显存选择上限，比较CPU多进程与
集中GPU批量推理吞吐；预热只使用验证片段，采样预算最多60秒，不按成功率选设备。
同观测、同随机数的CPU/GPU动作不一致时，auto回退CPU；显式CUDA不静默回退。

默认正式规模：主对比17,496局；E4验证加测试35,424局；E5报告97,200局；
E6报告30,000局；两个长模型扰动报告2,000局。这些是各报告的逻辑局数，
相同模型/环境/场景/采样任务跨报告共享。模型权重不同、标称配置一致时预计独立评估
169,240局，最终以实际任务键去重为准，不将共享引用再次计作新评估。

## 历史2M模型单独对比

原simulation_experiments/formal内6种方法各5seed的历史模型，可统一与75M模型比较：

```bash
python scripts/reevaluate_historical_2m.py run \
  --sealed-source results/remaining_formal_sources_20261027 \
  --workers auto --device auto
```

默认输出results/reevaluation_stochastic_v2/historical_2m_vs_75m，共31模型、20,088局。
原五类对照与75M的16,848局直接校验复用，正常仅补原有DR主动GRU的3,240局。
2M指训练预算，权重沿用原验证选出的selected_model，报告同时列出实际检查点步数。
此入口不训练，不替代800万E4。它暂停父派发器、等待在途E4任务完成，评估结束或异常
退出后自动恢复派发。plan只列任务与未经轨迹核验的文件存在数，run才全面校验缓存。

主对比在共同648场景配对，对照先对五个真实训练seed取场景均值。
75M和15M各是一份真实模型，不能复制成五个seed。采样重复先在模型/场景内取均值，
不增加独立训练次数。区间与结论须注明冻结模型、训练预算和训练物理差异；E3原始学习
曲线保持原训练验证口径。验收检查覆盖、完整性和可复现性，不以达到某个成功率为标准。
