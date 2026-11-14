# 办公场地批量对比

当前新实验默认采用 `small_office_v1`，地图与独立Mobile训练一致，气源/机器人全场
随机初始化，包含走廊和前厅。配置为 `configs/office_experiments.yaml`，基础环境为
`configs/mobile_rooms_hybrid_history.yaml`。上一版办公批量配置为
`configs/office_experiments_legacy.yaml`，使用 `configs/mobile_rooms_hybrid.yaml`。
开放场历史配置为 `configs/red_experiments.yaml` 及 `configs/e4_smoke.yaml` / `configs/e4_formal.yaml`。

## 入口与输出

| 入口 | 默认内容 | 默认输出根目录 |
|---|---|---|
| scripts/run_red_experiments.py | 办公场地E3/E4/E5/E6，省略实验选择时运行全部 | results/office_history_experiments |
| scripts/run_e4_experiments.py | 办公场地M/B0–B6主方法和基线对比 | results/office_history_e4 |
| scripts/run_e4_formal_conda.sh | 办公E4正式计划，沿用断点恢复机制 | results/office_history_e4 |

```powershell
# 查看计划，不启动训练
python scripts/run_red_experiments.py plan --profile smoke
python scripts/run_e4_experiments.py prepare --profile smoke --dry-run

# 先准备同一份验证/测试题库，再运行需要的实验
python scripts/run_red_experiments.py prepare --profile smoke
python scripts/run_red_experiments.py run --profile smoke --experiments e3 e5 --variants main,fixed,periodic,single_frame,history_mlp,single_left,single_right --resume

# 故障复评使用同一批已完成模型；模型与题库、签名必须匹配
python scripts/run_red_experiments.py robustness --profile smoke --variants main,fixed,single_left --faults nominal,delay2,gain80 --limit 1 --resume

# E4单独执行，也可由上面的完整实验入口委托
python scripts/run_e4_experiments.py run --profile smoke --methods main,b0_random,b1_reactive,b2_fixed,b3_scan,b4_single_frame,b5_history_mlp,b6_left,b6_right --resume
```

正式训练需显式使用 `--profile formal` 并核对plan预算。办公原生批量默认每配置
200万步、5个种子、8个环境；办公E4独立配置仍沿用其每方法100万步默认，可通过
`--timesteps` 明确统一预算。smoke原生每配置64步、1个验证/2个测试场景；所有默认
回合上限均为1200步。快速连通性验收可显式传 `--max-steps 12 --timesteps 32`，
缩短后的结果仅作软件测试，不能用于论文效果结论。

## 方法与公平性

所有方法共享办公墙体、碰撞、障碍感知动态puff、传感器、雷达和路径势差。
主方法默认用GRU处理19维气味、触须、风向和相对运动历史，再与当前42维导航分支融合。
MLP和单帧对照使用相同的导航分支。E5 特征组加入三项同结构对照：

| 对照 | 编码器内的处理 |
|---|---|
| current_wind_only | 用当前风向填充整个历史窗口 |
| no_motion_history | 相对位姿统一为零位移与零转角 |
| no_absolute_xy | 屏蔽导航分支的地图坐标，历史分支仍能读取相对位移 |

可用 `--experiments e5 --variants main,current_wind_only,no_motion_history,no_absolute_xy`
选择这组对照。各组配对复用训练种子、验证与测试场景，并报告命中、重捕获、到源与碰撞指标。

| 方法 | 策略输出 | 触须行为 |
|---|---|---|
| main / history_mlp / single_frame | 两个连续速度＋两个分类分支 | 左右独立选择 |
| fixed / b2_fixed | 两个连续速度 | 两侧固定扇区 |
| periodic / b3_scan | 两个连续速度 | 沿用周期往返扫描 |
| single_left / single_right | 两个连续速度＋一个分类分支 | 仅活动侧被策略控制，另一侧固定且信号屏蔽 |
| b0_random | 同主方法的合法混合随机动作 | 随机扇区 |
| b1_reactive | 气味规则＋雷达近障转向 | 规则扫描，不读取气源位置 |

PPO的概率为二维高斯与实际存在的分类分支相乘，不对高斯扇区取整。固定/周期方法
没有无效触须策略头；单触须只有一个分类头。所有速度在故障延迟和日志中保留小数。
三维或二维政策动作由包装器转换为共同四维环境命令，停驻编码为[-1,0,...]。

新场景直接封存环境的全场采样，替代无障碍场的下风侧摆点；验证与测试种子分离，
同场景在所有方法间配对复用。场景、代码、有效配置、地图摘要及各方法动作契约
写入manifest/模型/checkpoint，缓存与续训拒绝地图或配置混用。
默认首次无遮挡进入0.1米源范围即成功，未启用额外域随机化；with_dr及随机化敏感性
组显式打开扰动。no_blank_age对应新字段16；稀疏奖励消融也关闭房间路径势差正项。
导航汇总含碰撞、实际圆弧行程、路径效率、速度、近障比例及同区域/跨区域分组。

## 历史实验复现

```powershell
python scripts/run_red_experiments.py plan --config configs/red_experiments.yaml --profile formal
python scripts/run_red_experiments.py plan --config configs/office_experiments_legacy.yaml --profile formal
python scripts/run_e4_experiments.py status --config configs/e4_formal.yaml --profile formal --output-dir results/e4
```

`run_fast_comparison.py`、`run_remaining_formal.py`及其Conda启动器用于既有封存模型/
轨迹的专门复评，不会将旧模型、旧题库或历史成绩改标为新办公环境。它们继续核对
其来源签名；新办公实验使用本页入口及独立目录。修改过物理或代码的正式历史运行
应使用原封存源码，不通过更改签名绕过检查。

新版历史分支的短训验收记录位于 `results/office_history_verification`。
