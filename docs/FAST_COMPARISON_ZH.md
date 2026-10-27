# 75M主方法快速对比

此入口只转换旧轨迹、评估缺失场景和汇总，不包含训练/校准。它与完整E3–E6入口独立。

## 推荐命令

服务器正式开机、CUDA可用后，在项目目录执行：

```bash
bash scripts/run_fast_comparison_conda.sh --background
```

同一命令也用于中断恢复。先查看计划可运行：

```bash
bash scripts/run_fast_comparison_conda.sh plan
bash scripts/run_fast_comparison_conda.sh status
```

当前无卡模式只运行plan/status，不启动正式评估。run的默认auto模式在没有CUDA时明确退出；本地短验证可以显式指定CPU。后台提交只代表任务已启动，需查看打印的日志确认初始化与最终状态。

## 数据与协议

- 默认模型：`results/models/mobile_whisker_gru_ppo_1.zip`。从ZIP头读取实际seed与步数，冻结SHA-256；本次服务器已确认75,001,248步。
- 默认来源：`results/simulation_experiments/formal`。固定触须、周期扫描、单帧MLP、历史MLP、无DR历史主方法，各5个seed、648个场景，共16,200局。
- 默认输出：`results/simulation_experiments/arrival_75m`。不覆盖原数据；原2M主方法也仍留在旧目录。
- 成功：机器人中心距离气源≤0.1m，即时结束。max_steps、dt、场地和羽流由旧manifest读取；只将保持时间设0、半径显式设0.1、评估额外DR关闭。
- 旧轨迹完整校验后，截取首次到源的前缀，重新计算时间、命中、重捕获、路径和奖励。不能只修改成功标签而保留到源后的读数。
- 只有缺失或损坏的场景会补评。源模型、题库或科学代码不一致时拒绝混用，而不是放宽旧的恢复规则。
- 正常新增648局75M单模型评估；旧结果转换不执行策略推理。没有新增正式训练，也不将同一模型复制为5个训练seed。

## 性能设置

默认`--workers auto --device auto --calibration-seconds 32`。先检测CPU亲和性、cgroup配额、可用内存和CUDA显存，再比较安全并行上限与一半并行度。

转换进程不加载Torch/PPO；GPU方案只在协调器加载一份模型，CPU worker批量推进两个环境slot。CPU方案每进程加载一次模型，各进程均限制为一个计算线程。仅使用一张选定的可见GPU；小GRU的瓶颈通常需要通过实际吞吐确认，若CPU方案更快则选择CPU。

性能预热使用独立验证场景短片段，不选模型、不按成功率选设备、不反复运行正式测试场景。采样预算最多60秒，不含进程初始化与小段验证的收尾时间。配置与测量保存在`execution.json`。CPU/批量CUDA样例动作不一致时，auto排除CUDA方案；显式CUDA模式报错。

可显式覆盖：

```bash
bash scripts/run_fast_comparison_conda.sh run --workers 12 --device cpu --calibration-seconds 0
bash scripts/run_fast_comparison_conda.sh run --workers 12 --device cuda:0 --calibration-seconds 0
```

显式并行度仍受CPU和内存安全上限检查；不同硬件可重复同一命令，已完成的缓存不因并行度或报表改动而失效。

## 结果与边界

`state.json`记录总状态，`repairs.json`列出需要补评的旧场景；每局JSON与压缩轨迹原子保存。异常或中断后重新run只补缺项；不自动删除仍存在的运行锁。确认原任务已结束后才处理残留锁。

完整六方法应为16,848局。报告保存在`reports/<UTC时间戳>/report.md`，最新索引为`latest_report.json`，包括逐局、逐seed、覆盖率、分层、配对差值表与PNG/PDF/SVG。

配对单位是共同场景；对照先在场景内对5个seed取均值。bootstrap区间条件于现有冻结模型，不是训练重复的算法级区间。报告保留历史训练条件、预算和选中checkpoint步数，不能将75M对2M的收益全部解释成触须算法优势。完整消融与扰动本轮延期。

运行`python scripts/run_fast_comparison.py summarize`可重新生成报告，不执行模型评估。
