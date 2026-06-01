# 双触须主动嗅觉强化学习项目进度汇总

当前项目已经完成了双触须主动嗅觉强化学习仿真框架的最小闭环，并开始进入硬件最小闭环搭建阶段。仿真部分已经实现二维气味羽流环境、差速机器人模型、左右触须采样模型、一阶慢响应气体传感器模型、固定触须 baseline、联合控制 PPO 训练、独立评估、TensorBoard 监控和多种可视化结果图。主环境动作空间为 `MultiDiscrete([6, 10, 10])`，分别对应机器人移动动作、左触须扇区和右触须扇区。

当前主控制算法为 Joint PPO。策略根据左右传感器响应、浓度差、浓度变化、hit rate、触须角度、相对风向、机器人朝向和上一动作等连续观测，同时输出机器人移动动作和左右触须采样扇区。固定触须 baseline 中，策略只控制机器人移动，触须按照 10 个扇区周期扫描。为了提高训练泛化能力，PPO 训练脚本已经支持多环境采样和默认随机 seed，每次训练会记录实际 seed 和配置。

训练和评估流程已经工程化。训练脚本支持 TensorBoard，常用观察曲线包括训练 reward、评估 reward、PPO value loss、policy gradient loss、entropy loss 和 approx KL。评估脚本除了输出成功率、平均回报、最终距离、路径长度等导航指标外，还加入了触须信息采集指标，例如左右扇区计数、扇区组合计数、原始浓度均值、传感器响应均值、左右传感器差异、气味羽流接触比例和连续失去气味持续时间。

可视化方面，项目已经能够生成气味场轨迹图、传感器响应曲线、触须扇区统计图、触须动作空间离散分区示意图和评估 GIF 动画。结果图已调整为蓝色低浓度、黄色高浓度，并保留 colorbar。GIF 动画已放慢播放速度，并显示左右触须扇区，便于观察触须行为。

当前算法阶段的主要认识是：Joint PPO 可以作为主方法，但如果要证明触须主动控制的独立贡献，还需要补充更公平的 PPO baseline，例如周期触须 PPO 和随机触须 PPO。当前 fixed-whisker baseline 仍包含 DQN 版本，后续需要进一步统一算法，避免方法对比中混入算法差异。

硬件阶段当前条件已经明确：控制板为 STM32F103C8T6，执行机构为两个舵机，传感器为两个 MQ-3 气体传感器，供电问题已经解决，硬件线路已经完成连接。项目下一阶段目标不是直接在真实硬件上训练强化学习策略，而是先完成硬件最小闭环，即验证 `PC Python -> STM32 -> 舵机 -> MQ-3 -> PC` 的控制和数据链路。

STM32CubeMX 已经配置并生成工程，源码目录为 `hardware/firmware/stm32_dual_whisker/whisker/`。当前采用 STM32CubeMX、STM32CubeIDE 和 HAL 库开发下位机固件。PC Python 程序作为上位机，只负责通过串口发送左右触须扇区命令；STM32 C 程序作为下位机，负责 UART 命令解析、TIM PWM 舵机控制、ADC 采样和串口返回。

当前硬件引脚分配为：左舵机 PWM 信号接 PA0/TIM2_CH1，右舵机 PWM 信号接 PA1/TIM2_CH2，左 MQ-3 模拟输出接 PA2/ADC1_IN2，右 MQ-3 模拟输出接 PA3/ADC1_IN3，USART1 使用 PA9 作为 TX、PA10 作为 RX。需要注意 STM32F103C8T6 的 ADC 输入不能超过 3.3V，因此 MQ-3 AO 输出进入 PA2/PA3 前必须确认电压安全，必要时需要分压或电平保护。

当前 `Core/Src/main.c` 已经写入最小闭环 HAL 代码。固件启动后会启动 TIM2 两路 PWM，等待 USART1 串口命令。PC 发送 `STEP left_sector right_sector`，例如 `STEP 3 7`。STM32 解析命令后控制左右舵机转到对应扇区，等待舵机稳定，读取两个 MQ-3 ADC 值，并返回 JSON 数据，例如 `{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}`。

TIM2 已调整为适合舵机控制的 50Hz PWM。当前配置为 `Prescaler = 7`、`Period = 19999`，在 8MHz 时钟配置下计数单位约为 1us，PWM 周期约为 20ms。舵机角度到 PWM 脉宽的映射采用 0.5ms 到 2.5ms 的常见范围。左右舵机由 TIM2 两个通道并行输出 PWM，可以认为是同时控制。

当前每次 `STEP` 的执行间隔主要由舵机稳定等待和 ADC 多次采样组成。固件中 `SERVO_SETTLE_MS = 500`，`ADC_SAMPLE_COUNT = 10`。因此每次操作大约需要 500ms 舵机稳定等待，加上左右 MQ-3 顺序采样约 100ms，STM32 端完整处理约 600ms。PC 端 smoke test 默认还有 0.2s 延时，因此实际扇区切换间隔约为 0.8s。

左右 MQ-3 当前采用顺序 ADC 采样，即先读取左传感器，再读取右传感器，两个读数之间有几十毫秒时间差。考虑到 MQ-3 响应速度本身较慢，这个差异在初步 smoke test 阶段可以接受。后续如果需要更接近同步采样，可以考虑 ADC 扫描模式配合 DMA。

PC 端已经准备好硬件上位机脚本，包括 `dual_whisker_rl/hardware/serial_client.py`、`dual_whisker_rl/hardware/hardware_logger.py`、`scripts/hardware_smoke_test.py` 和 `scripts/hardware_sector_scan.py`。硬件依赖记录在 `requirements-hardware.txt`。烧录 STM32 固件后，可以运行 `python scripts\hardware_smoke_test.py --port COM3`，依次发送 `STEP 0 0` 到 `STEP 9 9`，保存 CSV 并绘制左右 MQ-3 ADC 响应曲线。

当前待完成事项是：在 STM32CubeIDE 中编译并烧录固件，使用 PC 上位机脚本测试串口通信、舵机转动和 MQ-3 数据返回。如果 smoke test 跑通，再进行周期扇区扫描、MQ-3 响应曲线记录、基线漂移观察和传感器简单标定。硬件链路稳定后，再考虑加载仿真训练好的 PPO 模型，仅执行左右触须扇区动作，观察策略在硬件上的采样行为。
