# STM32F103C8T6 双触须硬件接线说明

> 本文件保留开发板/飞线阶段的接线参考。自研 PCB 必须以 [PCB 原理图接口规范](../../docs/PCB_SCHEMATIC_INTERFACE_SPEC.md) 为唯一依据；若两者冲突，以该规范为准。

当前硬件：

- STM32F103C8T6；
- 左触须舵机；
- 右触须舵机；
- 左 MQ-3 气体传感器；
- 右 MQ-3 气体传感器；
- PC 通过开发板外接 USB-TTL，或通过自研板板载 USB-C/CH340C 与 STM32 通信。

当前目标是先完成最小 smoke test：

```text
PC Python 串口命令
-> STM32F103C8T6
-> 两个舵机转到指定扇区
-> 两个 MQ-3 ADC 采样
-> 串口返回采样值
```

## 推荐引脚分配

第一版推荐使用以下引脚：

```text
左舵机 PWM 信号  -> PA0  / TIM2_CH1
右舵机 PWM 信号  -> PA1  / TIM2_CH2

左 MQ-3 AO       -> PA2  / ADC1_IN2
右 MQ-3 AO       -> PA3  / ADC1_IN3

USART1_TX        -> PA9
USART1_RX        -> PA10
GND              -> 所有模块共地
```

## 舵机接线

```text
左舵机 Signal -> STM32 PA0
右舵机 Signal -> STM32 PA1

左舵机 VCC    -> 舵机外部供电正极
右舵机 VCC    -> 舵机外部供电正极

左舵机 GND    -> 舵机外部供电 GND
右舵机 GND    -> 舵机外部供电 GND
STM32 GND     -> 舵机外部供电 GND
```

注意：舵机建议使用独立电源供电，不建议直接从 STM32 板载 5V/3.3V 给舵机供电。STM32 与舵机电源必须共地，否则 PWM 控制信号可能不稳定。

## MQ-3 接线

```text
左 MQ-3 AO -> STM32 PA2 / ADC1_IN2
右 MQ-3 AO -> STM32 PA3 / ADC1_IN3

左 MQ-3 GND -> STM32 GND
右 MQ-3 GND -> STM32 GND
```

MQ-3 模块供电方式取决于你实际使用的模块。很多 MQ-3 模块使用 5V 供电，并且 AO 模拟输出可能接近 5V。

STM32F103C8T6 ADC 引脚最大输入电压不能超过 3.3V。因此必须确认：

```text
进入 PA2 / PA3 的模拟电压 <= 3.3V
```

如果 MQ-3 AO 最高可能超过 3.3V，需要在 AO 与 STM32 ADC 引脚之间加入分压电路或其他电平保护。

MQ-3 的 DO 数字输出第一阶段暂时不用，只使用 AO 模拟输出。

## 串口接线

### 开发板/飞线阶段

使用外部 USB-TTL 模块连接 USART1：

```text
USB-TTL TX -> STM32 PA10 / USART1_RX
USB-TTL RX -> STM32 PA9  / USART1_TX
USB-TTL GND -> STM32 GND
```

如果 STM32F103C8T6 开发板已经带 USB 转串口，则根据开发板实际串口连接选择对应 COM 口。

### 自研 PCB

自研板采用板载 USB-C + CH340C，不需要再外接 USB-TTL：

```text
USB-C D+/D- -> CH340C
CH340C TXD  -> STM32 PA10 / USART1_RX
CH340C RXD  <- STM32 PA9  / USART1_TX
```

USB VBUS 只给逻辑域供电，不能带舵机和 MQ-3。完整针序、电源或入和保护要求见 PCB 原理图接口规范。

## CubeMX 外设配置建议

```text
TIM2_CH1 -> PWM Generation CH1 -> PA0
TIM2_CH2 -> PWM Generation CH2 -> PA1

ADC1_IN2 -> PA2
ADC1_IN3 -> PA3

USART1_TX -> PA9
USART1_RX -> PA10
```

PWM 建议：

```text
PWM 频率：50 Hz
周期：20 ms
舵机 0 度脉宽：约 0.5 ms
舵机 90 度脉宽：约 1.5 ms
舵机 180 度脉宽：约 2.5 ms
```

第一阶段只要求舵机能按照扇区命令转动，MQ-3 能返回原始 ADC 值。滤波、标定、基线扣除和归一化后续再做。
