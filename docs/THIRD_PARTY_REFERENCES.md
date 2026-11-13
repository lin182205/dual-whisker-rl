# 连续避障导航参考

本项目的 Mobile 连续导航参考了 reiniscimurs 的
[DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation)，固定提交为
`1adfe511d6f3ff5d7d3bdc357914501c29646c39`。
参考项目采用 [MIT License（Copyright (c) 2021 reiniscimurs）](https://github.com/reiniscimurs/DRL-robot-navigation/blob/1adfe511d6f3ff5d7d3bdc357914501c29646c39/LICENSE)。

参考的代码位置：

- [velodyne_env.py](https://github.com/reiniscimurs/DRL-robot-navigation/blob/1adfe511d6f3ff5d7d3bdc357914501c29646c39/TD3/velodyne_env.py)：点云按角度分区取最小距离、线/角速度执行、近障成本。
- [train_velodyne_td3.py](https://github.com/reiniscimurs/DRL-robot-navigation/blob/1adfe511d6f3ff5d7d3bdc357914501c29646c39/TD3/train_velodyne_td3.py)：将归一化连续动作转换为非负线速度和有符号角速度。

本次在现有 Gymnasium/SB3 结构中独立实现混合 PPO、圆弧碰撞与雷达最小池化。
原项目是已知目标的 TD3 导航；本项目的气源未知，真实源点仅进入训练奖励和评估。
本项目使用360°扫描、36扇区、机器人表面间隙成本、碰撞后继续；保持主动双触须
和慢响应气味历史，分类扇区动作不由高斯取整得到。

训练由 PPO 原始采样动作计算联合概率；原参考中的近障随机动作覆盖不接入本项目，
以保持采样行为与 PPO 概率一致。底盘与嗅觉模型继续使用本仓库实现，无 ROS/Gazebo
运行依赖；本次没有复制或分发参考项目的源文件。

办公平面布局另参考Open-RMF Office World、Arena-Rosnav与TurtleBot3室内仿真，
具体来源、参考内容和独立设计坐标见[办公平面设计](OFFICE_LAYOUT_DESIGN.md)。
