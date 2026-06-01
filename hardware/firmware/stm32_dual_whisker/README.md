# STM32 双触须固件说明

本目录用于存放 STM32 双触须硬件控制固件。当前硬件阶段的目标是先完成最小 smoke test，不直接进行强化学习训练，也不做复杂传感器滤波或标定。

推荐工具链：

- STM32CubeMX；
- STM32CubeIDE；
- STM32 HAL 库。

## 上下位机分工

当前采用上下位机结构：

```text
PC Python 程序 = 上位机
STM32 C 程序 = 下位机固件
```

PC 端 Python 只负责通过串口发送触须扇区命令，并接收 STM32 返回的传感器数据。STM32 端 C 固件负责底层硬件控制，包括 UART 命令解析、定时器 PWM 舵机控制、ADC 采样和串口数据返回。

## 固件职责

STM32 固件需要实现：

- 接收串口命令，例如 `STEP left_sector right_sector`；
- 将左右触须扇区映射为舵机角度；
- 通过 PWM 控制两个舵机；
- 读取两个 MQ-3 传感器的 ADC 值；
- 通过串口返回一行 JSON 风格的数据。

## 串口协议

PC 发送命令：

```text
STEP 3 7
```

含义：

```text
左触须转到第 3 扇区
右触须转到第 7 扇区
```

STM32 返回：

```json
{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}
```

第一版只需要返回：

- `left_sector`：左触须扇区；
- `right_sector`：右触须扇区；
- `left_adc`：左 MQ-3 原始 ADC 值；
- `right_adc`：右 MQ-3 原始 ADC 值。

后续可以再增加：

- 时间戳；
- 左右舵机实际角度；
- 多次 ADC 平均值；
- 基线扣除值；
- 滤波值；
- 归一化浓度值。

## 外设分配建议

当前硬件板卡为 STM32F103C8T6。第一版推荐外设和引脚分配如下：

```text
左舵机 PWM 信号  -> PA0  / TIM2_CH1
右舵机 PWM 信号  -> PA1  / TIM2_CH2

左 MQ-3 AO       -> PA2  / ADC1_IN2
右 MQ-3 AO       -> PA3  / ADC1_IN3

USART1_TX        -> PA9
USART1_RX        -> PA10
```

CubeMX 中建议配置：

```text
TIM2_CH1 -> PWM Generation CH1
TIM2_CH2 -> PWM Generation CH2
ADC1_IN2
ADC1_IN3
USART1 Asynchronous
```

如果使用外部 USB-TTL 模块：

```text
USB-TTL TX -> STM32 PA10 / USART1_RX
USB-TTL RX -> STM32 PA9  / USART1_TX
USB-TTL GND -> STM32 GND
```

注意：STM32F103C8T6 的 ADC 输入电压不能超过 3.3V。很多 MQ-3 模块使用 5V 供电，AO 输出可能高于 3.3V，因此进入 PA2/PA3 前必须确认电压安全；必要时加入分压电路或电平保护。

## 舵机控制方式

舵机通常使用 50 Hz PWM 控制：

```text
PWM 周期：20 ms
0 度脉宽：约 0.5 ms
90 度脉宽：约 1.5 ms
180 度脉宽：约 2.5 ms
```

不同舵机的实际范围可能略有差异，后续需要通过校准修正。

当前触须动作空间为每侧 10 个扇区，粗略映射为：

```text
sector 0 -> 9 deg
sector 1 -> 27 deg
sector 2 -> 45 deg
...
sector 9 -> 171 deg
```

如果左右舵机安装方向相反，先不要在固件中写死复杂逻辑。建议先在 Python 配置或 STM32 固件中保留左右反向和角度微调参数，后续通过实际测试校准。

## 最小 HAL 实现逻辑

固件主流程可以按如下逻辑实现：

```text
1. 初始化 UART；
2. 初始化 TIM PWM 两个通道；
3. 初始化 ADC 两个通道；
4. 启动 PWM 输出；
5. 等待串口接收一行命令；
6. 解析 STEP left_sector right_sector；
7. 将 sector 转换为舵机角度；
8. 将舵机角度转换为 PWM compare 值；
9. 设置左右舵机 PWM；
10. 延时等待舵机稳定；
11. 读取左右 MQ-3 ADC；
12. 串口返回 JSON 风格数据。
```

伪代码示例：

```c
// UART RX line:
// STEP 3 7

int left_sector = 3;
int right_sector = 7;

float left_angle = sector_to_angle(left_sector);
float right_angle = sector_to_angle(right_sector);

uint16_t left_pwm = angle_to_pwm(left_angle);
uint16_t right_pwm = angle_to_pwm(right_angle);

__HAL_TIM_SET_COMPARE(&htimX, TIM_CHANNEL_1, left_pwm);
__HAL_TIM_SET_COMPARE(&htimX, TIM_CHANNEL_2, right_pwm);

HAL_Delay(settle_ms);

uint16_t left_adc = read_adc_left();
uint16_t right_adc = read_adc_right();

HAL_UART_Transmit(&huartX, response, strlen(response), HAL_MAX_DELAY);
```

## 第一阶段注意事项

第一阶段只追求链路跑通：

```text
PC 串口命令
-> STM32 接收
-> 舵机转动
-> MQ-3 ADC 读取
-> STM32 返回
-> PC 保存和画图
```

暂时不要加入：

- 复杂滤波；
- 基线漂移补偿；
- 传感器动态模型；
- PPO 策略推理；
- 底盘运动控制。

等 `STEP -> 舵机 -> ADC -> 串口返回` 稳定后，再逐步加入传感器标定、数据滤波和策略接入。

## CubeIDE 接入步骤

当前 CubeMX 生成工程位于：

```text
hardware/firmware/stm32_dual_whisker/whisker/
```

其中 `Core/Src/main.c` 已经写入最小闭环实现：启动 TIM2 两路 PWM，接收 `STEP left right` 串口命令，控制左右舵机，读取 PA2/PA3 两路 ADC，并返回 JSON 数据。

本目录提供了一个参考骨架：

```text
user_code_skeleton.c
```

它不是完整 CubeIDE 工程，而是用于复制到 STM32CubeIDE 自动生成工程的 `USER CODE` 区域中。

推荐流程：

```text
1. 用 STM32CubeMX 新建 STM32F103C8T6 工程；
2. 配置 TIM2_CH1 为 PA0 PWM；
3. 配置 TIM2_CH2 为 PA1 PWM；
4. 配置 ADC1_IN2 为 PA2；
5. 配置 ADC1_IN3 为 PA3；
6. 配置 USART1，PA9 为 TX，PA10 为 RX；
7. 生成 STM32CubeIDE 工程；
8. 在 main.c 的 USER CODE 区域加入 user_code_skeleton.c 中的函数；
9. 在初始化后调用 HAL_TIM_PWM_Start 启动两个 PWM 通道；
10. 在主循环中接收串口一行命令，并调用 handle_line(line)。
```

PC 端 smoke test 命令：

```powershell
python scripts\hardware_smoke_test.py --port COM3
```

如果串口号不是 COM3，可以改为实际串口号：

```powershell
python scripts\hardware_smoke_test.py --port COM5
```

重复扇区扫描：

```powershell
python scripts\hardware_sector_scan.py --port COM3 --cycles 5
```

输出结果默认保存在：

```text
results/hardware/
```
