# 空间记忆与空白区域重复访问筛选

本轮先比较“能否获得有效气味信息”，覆盖格数只作为行为诊断。A/B/C/D均使用
22.5厘米双触须、同一办公地图、同一羽流、0.2命中阈值、1200步回合上限和相同
三分支网络。四组都从零开始，训练种子11，各100万步；历史窗口20帧。

|组|空间记忆输入|首次访问奖励|空白惩罚|
|---|---|---|---|
|A|全零，保持同样网络容量|无|原绕圈扣分0.006|
|B|完整记忆|无|原绕圈扣分0.006|
|C|完整记忆|每新格最高0.02，整局预算0.5|原绕圈扣分0.006|
|D|完整记忆|同C|重复空白区域扣分0.006，替代原绕圈成本|

配置为`configs/mobile_rooms_spatial_{a,b,c,d}.yaml`。空间网格固定20×20、每格0.5米，
起始格记为已访问且不发奖。记忆只依赖机器人坐标和实际传感器阈值命中，不使用气源
坐标、真实浓度或未来羽流。近期空白证据60秒衰减、每格最高5秒；发生命中清除当前格
空白证据，因此不把间歇羽流的一次空白永久当作“这里没有气味”。

每次PPO更新采样4×1250=5000步，batch250；保存25万、50万和100万步检查点。
空间v2输入按57×20+800+2=1942维打包，不能从v1模型续训。训练和调度仍由现有训练
入口/管理面板负责，下面的入口只做封存、评估和报告，不启动任何训练。

## 先封存同一题库

```powershell
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py prepare `
  --output-dir results/spatial_exploration_pilot
```

开发集64局，种子420000–420063；保留集128局，种子520000–520127。两个文件写在
输出目录的`scenes/`下，含实际场景、场景SHA256、地图SHA256和物理摘要。生成场景时
统一用A组非物理设置，只允许空间输入开关、探索奖励、重复空白扣分及原绕圈扣分四项
实验差异共享题库；其它配置全部进入摘要。启动种子与是否记录轨迹不影响题库。
羽流、阈值、地图、速度或其它物理配置不同会拒绝复用；需要新协议时使用新的输出目录。
脚本不提供覆盖已有题库的开关。

## 每个检查点评估两种策略

下面模型路径仅示意，必须换成实际产物。`--checkpoint`必须等于模型内保存的累计
步数，可选250000、500000、1000000；同时核对模型训练种子为11。组别默认自动选择对应配置。

```powershell
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py evaluate `
  --variant A --checkpoint 1000000 `
  --model results/spatial_pilot_train/A/checkpoints/ppo_1000000_steps.zip `
  --split development --output-dir results/spatial_exploration_pilot
```

对B/C/D及三个检查点重复该调用。每次自动分别跑确定性策略和随机策略，每个场景各
运行一次，结果分别保存为JSON和逐回合CSV，包含模型、场景、物理、配置SHA256。
评估使用CPU、单推理线程、顺序执行；混合动作直接使用PPO原有的高斯+分类分布，
不经仅支持分类动作的旧随机采样工具。

## 自动等待检查点并完成筛选

服务器训练排队清单确定后，可在独立进程运行`watch`。部署生成的清单包含四组对应的
面板任务根目录，与面板数据库和任务管理分离；所有相对路径均相对于项目根目录：

```json
{
  "protocol_version": "spatial-exploration-pilot-v1",
  "groups": {
    "A": {"task_root": "results/training_dashboard/tasks/<A任务ID>"},
    "B": {"task_root": "results/training_dashboard/tasks/<B任务ID>"},
    "C": {"task_root": "results/training_dashboard/tasks/<C任务ID>"},
    "D": {"task_root": "results/training_dashboard/tasks/<D任务ID>"}
  }
}
```

各组默认使用对应`configs/mobile_rooms_spatial_<组别小写>.yaml`；也可以显式提供
`config`路径。评估器从任务下的`attempts/*/checkpoints/*_<步数>_steps.zip`查找检查点。
100万步检查点尚不存在时，仅允许使用实际保存步数恰为100万的
`attempts/*/final_model.zip`，暂停时未达预算的最终模型不会被误标成100万步。
所有找到的ZIP都会核对实际步数和写入完整性。

也支持独立训练的明确模型路径清单。下面展示一组，其余B/C/D必须补齐：

```json
{
  "protocol_version": "spatial-exploration-pilot-v1",
  "variants": {
    "A": {
      "config": "configs/mobile_rooms_spatial_a.yaml",
      "models": {
        "250000": "results/spatial_train/A/checkpoints/ppo_250000_steps.zip",
        "500000": "results/spatial_train/A/checkpoints/ppo_500000_steps.zip",
        "1000000": "results/spatial_train/A/checkpoints/ppo_1000000_steps.zip"
      }
    }
  }
}
```

可直接使用部署生成的完整清单。可先`prepare`，也可直接运行`watch`自动创建缺失题库：

```powershell
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py watch `
  --manifest results/spatial_exploration_pilot/evaluation_manifest.json `
  --output-dir results/spatial_exploration_pilot --poll-seconds 30
```

`watch`只读取ZIP已完整写入的检查点，逐一串行完成CPU评估；等待期间每30秒扫描一次，
已完成结果会复用。四组100万步开发集齐备后，自动按预设规则筛选，并只对A与最佳
候选做128局保留集复核。所有检查点和必要复核完成后自动退出，不创建、终止或延长
训练任务。`watch_status.json`记录待出现的模型路径及完成状态。若训练失败或暂停，
它会等待相应模型，应通过训练面板处理；`--once`只扫描一轮而不持续等待。

随机策略每回合开始前分别重置Python、NumPy和PyTorch随机数，动作种子由
`SHA256("spatial-exploration-pilot-v1/action/<场景种子>")`的前4字节按大端整数生成，
与模型路径和执行顺序无关。物理环境另用封存的场景种子重置。固定种子使同一运行环境下
随机复评可重复；更换PyTorch等依赖版本后不承诺逐动作相同。

已存在且签名完全一致的评估会复用；更换模型会保留另一个文件。报告拒绝同组、同
检查点、同题库、同策略出现多份结果，以免从多次模型中挑选。需要另一次实验时请使用
新输出目录。评估JSON/CSV和报告位于已忽略的`results/`中，不应提交模型和结果到Git。

## 指标与晋级

```powershell
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py report `
  --output-dir results/spatial_exploration_pilot
```

生成`report.md`、`report.json`及`report.csv`，按组别、检查点、题库与策略模式分列：

- 至少一次有效命中的回合数/比例（主指标）。
- 首次命中步数的1200步截尾均值及秒数，**全部回合均进入分母**；未命中或提前失败
  都计1200步，不能只对命中回合求均值。
- 到源率、碰撞步率/回合率、独立访问格数（包括起始格）。
- 首次访问奖励总额、重复空白成本总额/激活时长、原绕圈诊断比例、重新捕获次数。

晋级只依据100万步**开发集确定性策略**。必须先有完整四组结果，候选相对A至少
增加4/64个命中回合；多个候选优先命中回合数更多，再优先全部回合的首次命中截尾
均值更低，仍相同按组名字母顺序。25万/50万步及随机策略结果用于了解趋势，不改变
此规则。没有组达标就报告无候选，不自动追加步数。

确定最佳候选后，仅将候选与A的同一100万步模型在保留集复核：

```powershell
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py evaluate `
  --variant A --checkpoint 1000000 --model <A组模型路径> `
  --split heldout --output-dir results/spatial_exploration_pilot
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py evaluate `
  --variant D --checkpoint 1000000 --model <实际晋级组模型路径> `
  --split heldout --output-dir results/spatial_exploration_pilot
.\.venv\Scripts\python.exe scripts/run_spatial_exploration_pilot.py report `
  --output-dir results/spatial_exploration_pilot
```

示例D应替换成实际选中的组别。报告核对保留集模型SHA与开发集被选中的模型相同，
并分别列出两种策略模式的净新增命中、获得命中及丢失命中回合数。保留集不用于重新选组。

这只是**单训练种子的探索性筛选**，不等于稳定性能结论；增加覆盖格数也不自动代表
获得更多气味信息。四组触须都为22.5厘米，本轮不能归因主动触须或触须加长。若候选
通过保留集复核，再单独决定是否补训练种子、追加训练及开展固定/周期扫描公平对照。
