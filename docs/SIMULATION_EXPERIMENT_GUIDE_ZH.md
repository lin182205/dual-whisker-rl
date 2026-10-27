# E3–E6 仿真实验脚本中文使用指南

当前75M主方法主对比请使用[快速对比指南](FAST_COMPARISON_ZH.md)：进入0.1米半径即成功，转换旧轨迹并只补缺项，不启动完整E3–E6训练。下文完整实验协议继续保留，历史结果不覆盖。

本文对应统一入口 `scripts/run_red_experiments.py`。当前仅运行软件仿真，不需要串口、STM32或真实气体设备。

## 1. 最常用的命令

在仓库根目录执行以下命令，使用项目虚拟环境。未指定实验时默认依次运行 **E3、E4、E5、E6**；未指定阶段时默认 `run`；未指定规模时默认 `smoke`。

```powershell
# 只查看完整计划，不训练、不评估、不创建结果
.\.venv\Scripts\python.exe scripts\run_red_experiments.py plan

# 快速跑通全部实验；这不是论文正式结果
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --profile smoke --resume

# 仅跑E3和E6
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e3 e6 --profile smoke --resume

# 单独跑E5；--experiment是--experiments的别名
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiment e5 --profile smoke --resume

# 正式全部实验：包含大量独立训练，先用plan确认预算
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --profile formal --resume
```

`--experiments e3,e6` 与 `--experiments e3 e6` 等价，大小写均可；`all`代表全部。按E3→E4→E5→E6的固定顺序执行所选实验。未知或重复实验名会报错，不会悄悄改跑其他实验。

服务器已激活Python环境时，把 `.\.venv\Scripts\python.exe` 换为 `python` 即可。路径可以是相对仓库根目录的路径，也可以显式传绝对路径。

## 2. 四项实验分别做什么

| 编号 | 内容 | 默认配置 | 可以支持的结果 |
|---|---|---|---|
| E3 | 独立训练重复的学习曲线、观察区间AUC、后期波动与候选塌缩指标 | core六方法，formal每方法5seed | 学习是否稳定、是否依赖训练随机性 |
| E4 | 原E4九方法主对比、规则与固定角度校准 | 原运行器独立预算和场景协议 | 主方法与各基线在共享场景上的差异 |
| E5 | 历史、特征、奖励、几何、传感器、域随机化六类消融与敏感性 | 六组并集30种配置，分别重训 | 单一设计变化带来的配对效果 |
| E6 | 冻结策略后施加风、气源、传感器与执行器软件扰动 | core六方法，默认5条件含nominal | 绝对表现、相对nominal差值与比值 |

关注气味命中率、独立触须重捕获、响应趋势和左右信号差，同时报告成功率、距离、路径与失败惩罚时间。E5/E6图表优先展示命中率；表格保留相应任务和重捕获终点。

E3/E5/E6使用持续在气源半径内保持1秒的成功判据；原E4采用瞬时进入半径。两套结果在报告中分开，不能合并成功率。传入已有模型时，E4项采用当前持续1秒判据做单模型复评，报告会注明它不是完整九方法对比。

## 3. 已经有训练完成的模型

```powershell
# 不重新训练，只做已有模型普通评估与扰动测试
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 e6 --profile formal --model-path results/models/size_5-gru_3.zip --resume

# 不写--experiments也可：四项均进入流程，缺证据的项输出缺项状态表
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --profile smoke --model-path results/models/size_5-gru_3.zip --resume

# 只复评普通性能
.\.venv\Scripts\python.exe scripts\run_red_experiments.py evaluate --experiments e4 --profile formal --model-path results/models/size_5-gru_3.zip --resume
```

- 一个 `--model-path` 只对应一个模型、一个variant；未指定variant时使用 `main`。
- 其他方法显式传单个 `--variants history_mlp` 等，编码器、历史堆叠和动作接口必须匹配。
- 从模型读取真实训练seed和累计步数；formal的5seed不会把同一模型复制评估5次。
- 只有模型未保存seed时才用 `--model-seed` 声明真实seed，不能改写已有seed。
- 原文件只读，不继续训练，不创建假的训练完成状态或验证历史。
- 接口一致只说明可以调用；旧训练的场地、羽流、观测语义和传感器参数仍需核对。结果记录 `training_config_verified=false`。

**单个权重文件不能补齐E3/E5证据。** 没有训练日志时，E3输出 `missing_training_history` 状态表；没有各配置分别重训的模型时，E5输出 `missing_retrained_controls`。这两类表格用于记录证据缺口，不能当作论文实证结果。E4/E6仍正常复评已有模型。

如果确有该模型配套的原生验证日志，可以显式指定：

```powershell
.\.venv\Scripts\python.exe scripts\run_red_experiments.py summarize --experiments e3 --model-path results/models/size_5-gru_3.zip --learning-dir results/对应实验/formal --resume
```

日志根目录需包含 `<variant>/seed_<真实训练seed>/state.json`（状态为completed）和 `validation_history.json`。只读取模型实际seed；日志同源性仍需人工核对。当前不从TensorBoard事件文件重建已经缺失的验证场景数据。只有一次训练仍不能说明多seed稳定性。

## 4. 从头分别运行各项实验

### E3：学习与训练稳定性

```powershell
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e3 --profile formal --resume
```

默认六方法：main、fixed、periodic、single_frame、history_mlp、no_dr。每次验证保留成功率、命中率、重捕获、奖励分量与PPO诊断。学习曲线的区间按训练seed重采样，单seed不显示算法级CI；AUC只积分已有验证点覆盖的区间，不补0步表现。

### E4：主对比

```powershell
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 --profile formal --resume

# 快速只验证主方法和随机规则
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e4 --profile smoke --e4-methods main,b0_random --resume
```

原E4方法为main、b0_random、b1_reactive、b2_fixed、b3_scan、b4_single_frame、b5_history_mlp、b6_left、b6_right，其中7种需要训练，规则法不占独立训练seed。正式运行沿用原E4校准及默认每模型100万步，不能称为新增200万步对比。

`--e4-methods`只限制原E4的方法；`--group`与`--variants`只限制E3/E5/E6。传入 `--model-path` 后不委托原E4，按该模型对应variant单独复评。

### E5：消融与敏感性

```powershell
# 默认运行六类消融，共30种配置，重叠项复用
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --profile formal --resume

# 只做历史长度消融
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --group history --profile formal --resume

# 小范围验证两个配置
.\.venv\Scripts\python.exe scripts\run_red_experiments.py run --experiments e5 --variants main,h10 --profile smoke --resume
```

| group | 改变的因素 |
|---|---|
| history | H1/5/10/20/40及历史MLP |
| features | 趋势、显式差分、blank_age、角度、原始读数 |
| rewards | 去重捕获、最佳浓度、停滞惩罚和稀疏奖励配置 |
| geometry | 触须长度0/0.075/0.15/0.225m及3/5/10/15扇区 |
| sensor | 理想/标称动态×GRU/MLP与时间常数扫参 |
| randomization | 无、窄、默认、宽域随机化 |

除sensor组参考nominal_gru外，其他组参考main。表格中差值统一为配置减参考。需要对照分别训练；不将推理时删特征冒充训练消融。不跨奖励配置直接比较回报大小。

### E6：冻结模型扰动测试

```powershell
# 已有原生实验模型时，仅做评估，不新增训练
.\.venv\Scripts\python.exe scripts\run_red_experiments.py evaluate --experiments e6 --profile formal --variants main,no_dr --faults wind24,gain80,servo150,delay2 --resume

# 短检查；robustness是仅E6的兼容阶段名
.\.venv\Scripts\python.exe scripts\run_red_experiments.py robustness --profile smoke --variants main,no_dr --limit 2 --resume
```

默认条件为nominal、wind24、gain80、servo150、delay2。若没有在 `--faults` 中填写nominal，会自动加入，保证有同场景基线。formal默认每方法每seed每条件200个共享测试场景；`--limit`只改变E6场景数，不改变普通测试集。

模型冻结，故障不会用于挑选checkpoint或调整参数。差值为扰动减nominal；nominal为零时比值留空。单模型bootstrap区间只描述该模型在场景上的变化，不是算法级置信区间。

`run --experiments e6` 在没有原生模型时会按所选配置训练后再测试；需要严格禁止新增训练时使用 `evaluate --experiments e6` 或传 `--model-path`。

## 5. 阶段与参数速查

| 参数或阶段 | 含义 |
|---|---|
| plan / --dry-run | 查看所选实验、方法、训练任务及来源，不写结果 |
| prepare | 封存配置、代码和验证/测试场景，不执行实验 |
| run | 完成所选实验的必要训练、评估及图表报告 |
| train | 只训练，随后按已有数据导出报告；没有测试数据的项标明缺项 |
| evaluate | 不训练，执行所选评估；E3只分析已有日志 |
| summarize | 不训练、不评估，只按已保存数据重新生成独立报告 |
| robustness | 仅E6，不能与其他实验选择混用 |
| e4 | 原E4兼容入口，由--e4-stage指定阶段 |
| --profile smoke/formal | 连通性/正式预算，默认smoke |
| --experiments / --experiment | e3/e4/e5/e6，省略为全部 |
| --group | 限制E3/E5/E6；默认E3/E6为core，E5为六类全部 |
| --variants | 明确指定E3/E5/E6的配置，覆盖group |
| --e4-methods | 原E4方法列表，与原生variant分别设置 |
| --model-path / --model-seed | 已有模型及缺失训练seed的声明 |
| --learning-dir | 已有模型对应验证日志根目录 |
| --faults / --limit | E6条件列表及每条件场景数 |
| --timesteps / --n-envs / --max-steps | 显式覆盖训练预算、环境数及局长；timesteps和n-envs也传给原E4 |
| --bootstrap | 报告bootstrap次数，默认2000；短验证可降低，不建议用低次数作最终论文区间 |
| --output-dir | 结果根目录，自动附加profile |
| --resume | 校验后复用已完成模型、场景结果或继续未完成训练 |

未指定 `--group/--variants` 时，完整E3/E5/E6共享33种原生配置。formal共165次200万步训练；原E4另有35次100万步训练，以及正式校准开销。计划中的training_tasks不包含原E4校准候选训练。不要把8个环境进程当成8份独立训练预算。

## 6. 输出文件怎么用于论文

每次实验命令会创建新的UTC时间戳报告目录，即使复用评估缓存，也会重新导出图表，保留旧报告：

```text
results/simulation_experiments/<profile>/
  manifest.json
  scenarios_validation.json
  scenarios_test.json
  <variant>/seed_<seed>/...                 原生模型、日志、逐局数据
  e4/<profile>/...                         原E4独立协议数据
  reports/<UTC时间戳>/
    index.md                              本次结果索引
    experiment_status.csv                 所选实验完成/缺项/失败状态
    run_manifest.json
    e3/ e4/ e5/ e6/                       仅创建所选实验
      report.md                           中文结果说明及文件链接
      report_manifest.json                输入路径与SHA-256、统计和图形契约
      tables/*.csv
      figures/*.png                       600dpi预览
      figures/*.pdf                       可编辑矢量图
      figures/*.svg                       可编辑矢量图
      figures/*.md                        对应图注和区间定义
```

已有模型的默认根目录为 `results/simulation_experiments/pretrained/<模型名>/<profile>/`。指定 `--output-dir` 后仍自动附加profile。所有产物在results下，不提交模型、日志或图像到Git。

| 实验 | 主要表格 | 主要图片 |
|---|---|---|
| E3 | learning_curves、validation_points、training_seed_metrics、coverage | 成功率学习曲线、气味命中率学习曲线 |
| E4 | episodes、method_metrics、seed_metrics、coverage | 方法成功率、气味命中率点区间图 |
| E5 | episodes、paired_effects、coverage | 各消融组相对参考的配对效应图 |
| E6 | episodes、degradation、coverage | 成功率及命中率退化热图，条件多时分图 |

每项至少生成experiment_status.csv和report.md。缺少真实数值时保留空值并列出缺项，不能将缺项表当成论文效果数据。部分实验失败时，独立实验继续执行并各自记录状态，命令最终返回非零退出码；人为中断不自动继续其他实验。

图表是自动生成的实验结果材料，投稿前仍须检查最终尺寸、标签、统计假设、实验重复与期刊要求；不承诺特定期刊接收或格式合规。

## 7. 统计与复现注意事项

1. 先看coverage，未完成的场景保持缺失，不计为失败，也不填成零。
2. 训练seed才是训练重复。同一次训练的多个checkpoint不能冒充多个独立seed。
3. E3按seed bootstrap；E4绝对值采用等权seed均值及seed区间，规则法不展示算法级区间。
4. E5按共享场景和训练seed交叉重采样，少于2seed不画算法级区间；E6单模型条件区间单独标注。
5. E5的Holm只在每消融组每终点内校正。跨组/跨终点主要假设的检验族仍须在正式实验前约定。
6. 5seed双侧符号置换最小p=0.0625，不能以大量episode替代独立训练重复来制造显著性。
7. 没有重捕获事件时相关时间保持缺失；nominal接近零时不报告不稳定比值。
8. 软件故障不一定都是严格域外条件，需对照训练域随机化范围。
9. `smoke`只有1seed和很短预算，全部结果都只证明流程可以运行。
10. 每张图的原始统计表、逐局记录与来源摘要一起保留，不能只留下均值图片。

## 8. 断点恢复和常见问题

- **重复运行**：沿用相同参数并加 `--resume`。同一次完整命令中共享的原生模型自动去重；跨命令复用需--resume。
- **先E3后E5/E6**：同一物理配置、代码、预算和题库下可以沿用结果目录；实验选择不进入科学封存签名。
- **改预算、模型或配置**：使用新 `--output-dir`。本次新增调度不属于历史显示版本兼容例外，不放宽旧封存。
- **只有旧训练结果**：可用 `--model-path` 在新目录复评；不要复制完成状态伪装成新配置训练结果。
- **旧15维模型报错**：当前Mobile单帧为16维，需要匹配原环境或观测适配，不能通过随意填零绕过语义差异。
- **报告没有图**：先查看experiment_status.csv和coverage；数据缺失时脚本只输出真实状态表。
- **锁文件**：不要同时对同一profile运行多个训练/评估/汇总进程。强杀留下.run.lock时，确认原进程已结束后再处理。
- **进度显示**：普通/扰动评估有场景进度与剩余估时；这个倒计时不含其他实验的未来训练。
- **只想重新画图**：使用summarize并指定实验，新报告会进入新的时间戳目录，不改原始数据。

## 9. 开发验证入口

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment_suite.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pretrained_experiments.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment_progress.py -v
.\.venv\Scripts\python.exe scripts/check_portable_paths.py
```
