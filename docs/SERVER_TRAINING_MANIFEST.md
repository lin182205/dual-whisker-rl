# 服务器训练文件与依赖清单

本清单从现有训练、评测及 E4 统一实验入口的导入链整理。目标是让服务器只接收可运行实验所需内容，同时保留可复现配置和中断恢复能力。

## 1. Python 与系统依赖

正式 E4 训练和批量评测需要 Python 3.10 或更新版本。推荐一次安装：

```bash
python -m pip install -r requirements-server.txt
```

| 依赖 | 用途 | E4 正式实验 |
|---|---|---|
| `numpy` | 环境动力学、观测和指标计算 | 必需 |
| `gymnasium` | 环境与动作/观测空间接口 | 必需 |
| `PyYAML` | 读取实验配置 | 必需 |
| `torch` | PPO 网络、GRU/MLP 特征提取器 | 必需 |
| `stable-baselines3` | PPO/DQN、并行环境、checkpoint 和评估回调 | 必需 |
| `tensorboard` | 写入和查看训练曲线 | 必需 |
| `matplotlib` | 旧评测脚本的场图、曲线和动画 | E4 批量评测不用；现有基础依赖会安装 |
| `pyserial` | PC 与 STM32 串口通信 | 仿真不需要 |
| FFmpeg | H.264 MP4 动画编码 | 批量训练/评测不需要；导出动画时单独安装 |

`requirements-server.txt` 汇总了 `requirements.txt` 与 `requirements-rl.txt`。真机实验才需要另装 `requirements-hardware.txt`。

## 2. E4 正式实验最小文件集

以下内容必须上传：

- `dual_whisker_rl/`：环境、羽流、传感器、观测、GRU/MLP 策略适配和 E4 业务逻辑。
- `scripts/run_e4_experiments.py`：`prepare/calibrate/train/evaluate/summarize/run/status` 统一入口。
- `configs/e4_formal.yaml`：正式场景、场地、并行环境和训练配置。
- `configs/e4_smoke.yaml`：服务器正式开跑前的短链路检查。
- `requirements.txt`、`requirements-rl.txt`、`requirements-server.txt`：Python 依赖。

推荐同时上传：

- `README.md`、`PROJECT_RECORD.md`：运行命令、实验口径和变更记录。
- `scripts/check_portable_paths.py`：上传前检查机器相关绝对路径。
- `tests/test_e4_experiments.py`：服务器环境安装后可选的 E4 快速回归检查。

训练生成的模型、checkpoint、TensorBoard 和逐场景评测结果全部写入 `results/`。该目录不进入 Git，但中断恢复需要它；更换服务器或释放实例前必须单独同步。

## 3. 正式训练不需要上传的内容

| 路径 | 内容 | 处理建议 |
|---|---|---|
| `.venv/` | Windows 本地虚拟环境，当前约 729 MB | 排除，在服务器重建 |
| `results/` | 模型、日志、动画和历史实验，当前约 1.6 GB | 首次上传排除；续训时单独同步目标实验目录 |
| `hardware/` | STM32 固件、接线和真机工具 | 纯仿真训练排除 |
| `hardware/firmware/**/build/` | STM32/CMake 本机构建缓存 | 始终排除 |
| `docs/` | 硬件、算法图和说明文档 | 运行不需要；保留本清单也可只单独上传 |
| `active-olfaction-pape/` | 已从主仓库移除的本地独立论文工作区 | 排除；需要时从论文仓库单独管理 |
| `tmp/` | PDF 提取、参考资料和临时工具，当前约 33 MB | 排除 |
| `output/` | 文档/PDF 输出 | 排除 |
| `figures/`、`scripts/figures/` | 图形资源和图表脚本资源 | 不生成图时排除 |
| `tests/` | 回归与 smoke 测试 | 正式训练运行不需要；建议部署验收后再排除 |
| `.git/` | Git 历史与对象库 | 直接归档上传时排除；用 `git clone` 时自然保留 |
| `.vscode/`、`.idea/`、`.vs/`、`.claude/`、`.codex/`、`.agents/` | 本地 IDE 与助手状态 | 排除 |
| `scripts/generate_*`、`scripts/visualize_*`、`scripts/analyze_*` | 画图、动画和诊断入口 | 正式批量实验不需要 |
| `scripts/hardware_*`、`scripts/compare_hardware_*` | 真机采集与分析入口 | 纯仿真训练排除 |

不要整体排除 `scripts/`：E4 入口位于该目录。若需要旧训练/评测入口，应连同对应脚本一起上传；其中移动评测脚本还会调用 `train_whisker_only_ppo.py` 和 `visualize_mobile_whisker_env.py`。

## 4. 服务器启动顺序

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
python scripts/run_e4_experiments.py prepare --profile smoke
python scripts/run_e4_experiments.py run --profile smoke --resume
python scripts/run_e4_experiments.py status --profile smoke
python scripts/run_e4_experiments.py calibrate --profile formal
python scripts/run_e4_experiments.py run --profile formal --resume
```

正式训练默认使用 8 个 `SubprocVecEnv` worker。若服务器 CPU 核数或内存不足，可通过 `--n-envs` 调低；GPU 主要负责较小的 PPO 网络，环境羽流计算和多进程采样通常更依赖 CPU。

## 5. 上传与恢复检查

上传前：

```powershell
.\.venv\Scripts\python.exe scripts\check_portable_paths.py
git status --short
```

服务器启动后先完成 smoke，再开启 formal。恢复训练时保留同一输出目录（默认 `results/e4/`）并添加 `--resume`；运行器会核对版本、配置、场景摘要和代码摘要，不兼容的产物不会被静默复用。smoke 与 formal 若需要长期并存，应分别通过 `--output-dir results/e4_smoke` 和 `--output-dir results/e4_formal` 指定目录，避免共用 manifest。
