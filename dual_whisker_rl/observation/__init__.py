"""共享观测构造模块。

仿真环境和硬件 rollout 脚本共用同一套 observation builder，保证训练时
和真机部署时构造的观测语义一致（sim-to-real 观测对齐）。
"""

from dual_whisker_rl.observation.whisker_observation_builder import (
    WhiskerObservationBuilder,
)

__all__ = ["WhiskerObservationBuilder"]
