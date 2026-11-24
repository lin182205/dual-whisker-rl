# 服务器训练管理面板

面板运行在训练服务器，只监听本机地址。浏览器通过 SSH 转发访问，关闭网页和
重启面板服务都不会结束独立运行的训练进程。SSH 密码不写入面板或脚本。

## 打开面板

Windows 项目根目录执行：

```powershell
.\scripts\open_training_dashboard.ps1
```

已有转发会直接复用。否则在新 SSH 窗口输入服务器密码，面板就绪后自动打开
`http://127.0.0.1:8866`。也可自行保持下面的连接，再打开该地址：

```bash
ssh -N -L 8866:127.0.0.1:8866 -p 28306 root@connect.nmb2.seetacloud.com
```

服务器迁移后可给 PowerShell 脚本指定 `-SshPort`。本机8866被其他服务占用时，
指定 `-LocalPort 8867`；无需更改服务器面板端口。

## 任务与进度口径

- 当前两组仅接入监控。原版从1500万步追加2000万步，累计目标3500万步；
  改进版从零训练2000万步。上线不会中断或重启它们。
- “本次完成”按任务起始步数到目标的区间计算；“累计进度”包含原模型已有步数。
  暂停恢复不会重置任务起点。接管发生回退时会记录来源和检查点。
- 创建任务默认进入 FIFO 队列，最多同时运行两组。暂停、结束且进程完全退出后，
  才释放名额。恢复任务也进入队列。从零默认随机种子，续训默认沿用源任务种子。
- 续训通过界面选择已登记的检查点，沿用环境、编码器、并行环境数和PPO参数。
  不开放任意命令、文件上传或运行中修改奖励和环境动力学。
- “在当前目标上追加500万步”会将3500万目标改为4000万，而非按当前进度重算。
  目标向上对齐一整轮采样步数，界面返回实际目标；低于当前步数请使用停止。

曲线来自 TensorBoard 训练日志。成功率明确标注最近500回合，不代表独立评估结果。
旧版任务没有绕圈比例时显示“未记录”，不会填零。进度优先使用完整 PPO 更新后的
轮次结束记录，不能把启动时写一次的metadata当作实时进度。估时至少积累3轮，
按最近5轮计算；长时间日志停更提示异常，不自动重启训练。

## 暂停、继续、停止和接管

管理进程通过原子写入的控制文件向独立训练进程发请求。训练在下一轮采样前，
即上一轮梯度更新完成后处理请求。暂停或停止时先原子保存包含优化器的模型和
检查点，再关闭环境子进程并退出。暂停中不占用GPU。继续会重置仿真回合，
保留模型、优化器和累计步数，不承诺逐样本复现暂停前的环境状态。

“停止”是终态，可从其检查点另建续训任务；“暂停”可以继续原任务。
不能通过暂停或停止未接管的旧任务来绕过接管步骤。

接管现有任务需要点击“接管控制”，先查看检查点与预计重新采样步数，再确认。
管理服务先冻结检查点副本并执行模型/环境预检，成功后才向已校验身份的原进程组
发送停止信号，再排队启动新的运行段。原日志、模型和配置保留。无完整检查点或
模型不兼容时不会停止旧进程。接管后曲线只保留父运行段截至来源检查点的数据，
不把已丢弃的运行进度当成继续训练的数据。

## 服务部署

将新版源码部署到独立的管理目录，保留原版与改进版正在运行的源码目录。
下面目录是本次实例的部署约定，不是代码中的默认服务器路径。

```bash
cd /root/autodl-tmp/dual-whisker-rl-manager
/root/miniconda3/envs/dual-whisker-rl/bin/python -m venv /root/autodl-tmp/dual-whisker-dashboard-venv
/root/autodl-tmp/dual-whisker-dashboard-venv/bin/python -m pip install -r requirements-dashboard.txt

/root/autodl-tmp/dual-whisker-dashboard-venv/bin/python -m dual_whisker_rl.dashboard \
  --directory results/training_dashboard \
  --training-python /root/miniconda3/envs/dual-whisker-rl/bin/python \
  --import-manifest /root/autodl-tmp/dual-whisker-rl-comparison/results/office_pair_20m_20261124_122126/launch_manifest.json
```

正式运行时把最后一条命令放入名为 `training-dashboard` 的独立tmux会话，并将输出
重定向到管理目录的results中。管理服务固定为单个Uvicorn worker，并对管理目录
加单实例锁。新训练进程使用原来的训练Python，Web进程不导入torch或模型。

状态位于 `results/training_dashboard/dashboard.sqlite`，每个任务及运行段另存
控制文件、状态文件、配置、模型、检查点、日志和启动身份记录。该目录已被
results忽略规则排除，不提交模型、日志、数据库或密码。重新运行相同导入命令
不会重复创建任务，服务重启不会重新派发仍存活的进程。

只用CPU做本地隔离验证时可加 `--allow-cpu`。服务器默认要求GPU可用，失败时不会
悄悄改成CPU长训。暂停/停止故障或进程状态无法核实时，请查看任务日志和服务器
进程；面板不会自动强杀未知进程或删除训练产物。

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt httpx
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_training_dashboard*.py' -v
```

包含日志/路径/缺失数据检查、CSRF与参数边界、重复提交、FIFO与恢复，
以及在真实Mobile环境中完成双子进程短训、暂停、管理服务重建和检查点续训。
模型级测试确认无控制请求时与普通PPO更新结果一致，并验证延长目标、缩短目标和
优化器恢复。测试使用独立临时目录，不导入或控制服务器正式任务。
