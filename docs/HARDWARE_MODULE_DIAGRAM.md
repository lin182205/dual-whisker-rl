# 双触须主动嗅觉硬件架构草图

本图仅描述论文中的真实硬件闭环：上位机下发左右触须控制信号，STM32 驱动双舵机改变采样位置，双 MQ-3 完成气体采样，STM32 将左右 ADC 数据回传上位机。数据预处理、实验日志、指标计算和强化学习网络不属于本硬件架构图。

> 本文件是概念架构说明。自研 PCB 的连接器针序、电源树、分压参数、最小系统、网络命名和审图要求，以 [PCB 原理图接口规范](PCB_SCHEMATIC_INTERFACE_SPEC.md) 为唯一依据。

## 硬件控制与气体采集闭环

```mermaid
flowchart LR
    PC[PC 上位机<br/>触须控制指令生成<br/>串口命令发送与采样结果接收]
    DOWN[下行串口<br/>STEP 左扇区 右扇区]
    LINK[板载 USB-C / CH340C<br/>USART1，115200 baud、8N1]

    subgraph MCU[STM32F103C8T6 控制器]
        direction TB
        PARSE[解析左右扇区<br/>范围 0 至 9]
        MAP[扇区映射为舵机角度]
        PWM[产生两路 50 Hz PWM]
        WAIT[等待舵机稳定 500 ms]
        ADC[读取左右 ADC<br/>每通道 10 次采样取平均]
        PACK[封装左右扇区与 ADC 数据]

        PARSE --> MAP --> PWM --> WAIT --> ADC --> PACK
    end

    subgraph WHISKERS[双触须执行与气体感知]
        direction TB
        SERVOS[左、右舵机]
        POS[左、右触须采样位置]
        GAS[气源、气流与局部气体浓度]
        SENSORS[左、右 MQ-3]
        PROTECT[分压 / 电平保护<br/>ADC 输入不超过 3.3 V]

        SERVOS -->|改变角度| POS
        POS -->|确定空间采样点| SENSORS
        GAS -->|局部气体刺激| SENSORS
        SENSORS -->|左右 AO 模拟电压| PROTECT
    end

    UP[上行串口 JSON<br/>左扇区、右扇区<br/>左 ADC、右 ADC]

    PC --> DOWN --> LINK --> PARSE
    PWM -->|TIM2 两路 PWM| SERVOS
    PROTECT -->|PA2、PA3 模拟输入| ADC
    PACK --> UP --> LINK -->|本步气体采样结果| PC
    PC -->|依据当前控制方式产生下一步指令| DOWN

    style PC stroke-width:5px,font-size:20px
```

## 接口协议

上位机每次发送一条左右触须扇区命令：

```text
STEP <left_sector> <right_sector>\n
```

左右扇区均为整数 `0..9`。例如：

```text
STEP 3 7
```

STM32 完成舵机转动和双通道 ADC 采样后，返回一行 JSON：

```json
{"left_sector":3,"right_sector":7,"left_adc":1820,"right_adc":1765}
```

串口采用 USART1，参数为 `115200 baud、8 data bits、1 stop bit、no parity`。一次命令对应一次响应；上位机收到响应后才进入下一次控制与采样循环。

## 关键硬件接口

| 功能 | STM32 接口 | 信号方向 |
|---|---|---|
| 左舵机控制 | PA0 / TIM2 CH1 | STM32 → 左舵机 PWM |
| 右舵机控制 | PA1 / TIM2 CH2 | STM32 → 右舵机 PWM |
| 左 MQ-3 采样 | PA2 / ADC1 IN2 | 左 MQ-3 AO → STM32 ADC |
| 右 MQ-3 采样 | PA3 / ADC1 IN3 | 右 MQ-3 AO → STM32 ADC |
| 上位机通信 | PA9 TX、PA10 RX / USART1 | STM32 ↔ 板载 CH340C ↔ USB-C |
| SWD 下载调试 | PA13 SWDIO、PA14 SWCLK、NRST | ST-Link ↔ STM32 |

自研板使用外部稳压 5V（建议能力不低于 3A）给舵机和 MQ-3 供电；USB VBUS 仅可给 MCU/CH340C 逻辑域供电。两路 MQ-3 AO 必须先经过规范规定的分压滤波，再进入 PA2/PA3，禁止直接连接。

## 论文图注建议

> 双触须主动嗅觉硬件闭环。PC 上位机通过串口下发左右触须扇区指令，STM32 将扇区映射为两路舵机 PWM，使左右 MQ-3 在不同空间位置采样。舵机稳定后，STM32 对两路 MQ-3 模拟信号进行 ADC 采样和平均，并将扇区及 ADC 读数通过 JSON 串口报文返回上位机，形成“控制触须位置—采集气体响应—回传测量数据”的循环。
