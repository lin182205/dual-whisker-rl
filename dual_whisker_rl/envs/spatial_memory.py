"""只依赖机器人定位与实际气味命中的回合空间记忆。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SPATIAL_MEMORY_LAYOUT = "wind_motion_spatial_memory_v2"
CORE_OBSERVATION_DIM = 57
GRID_SIDE = 20
GRID_CELL_SIZE_M = 0.5
MAP_OBSERVATION_DIM = GRID_SIDE * GRID_SIDE * 2
SPATIAL_OBSERVATION_DIM = MAP_OBSERVATION_DIM + 2
RAW_OBSERVATION_DIM = CORE_OBSERVATION_DIM + SPATIAL_OBSERVATION_DIM


class EpisodicSpatialMemory:
    """记录首次访问及近期无命中暴露，不推断永久无气味区域。

    地图以 ``[行y, 列x]`` 存储。负坐标从世界下界 -5 米起做 floor；
    边界及越界终止位置夹到最近边缘格，仅用于记录，不改变环境碰撞或终止。
    """

    blank_decay_tau_s = 60.0
    blank_cap_s = 5.0
    no_new_age_cap_s = 5.0
    blank_reward_delay_s = 5.0
    revisit_start_s = 2.0
    revisit_ramp_s = 3.0
    budget_normalizer = 0.5

    def __init__(
        self,
        *,
        exploration_bonus: float = 0.0,
        exploration_budget: float = 0.5,
        blank_revisit_penalty: float = 0.0,
    ) -> None:
        for name, value in {
            "exploration_bonus": exploration_bonus,
            "exploration_budget": exploration_budget,
            "blank_revisit_penalty": blank_revisit_penalty,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须为有限非负数")
        if exploration_budget > self.budget_normalizer:
            raise ValueError("exploration_budget 不得超过 0.5")
        self.exploration_bonus = float(exploration_bonus)
        self.exploration_budget = float(exploration_budget)
        self.blank_revisit_penalty = float(blank_revisit_penalty)
        self.visited = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.bool_)
        self.blank_exposure_s = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.float64)
        self.reset((0.0, 0.0))

    @property
    def V(self) -> np.ndarray:
        """供可视化读取的回合访问图，行y、列x。"""
        return self.visited

    @property
    def B(self) -> np.ndarray:
        """供可视化读取的近期空白暴露图，单位秒。"""
        return self.blank_exposure_s

    @staticmethod
    def cell_index(position: tuple[float, float]) -> tuple[int, int]:
        """返回行、列索引；精确格线归入较大的索引。"""
        x, y = position
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("空间记忆坐标必须为有限数")
        column = min(GRID_SIDE - 1, max(0, math.floor((x + 5.0) / GRID_CELL_SIZE_M)))
        row = min(GRID_SIDE - 1, max(0, math.floor((y + 5.0) / GRID_CELL_SIZE_M)))
        return row, column

    def reset(self, position: tuple[float, float]) -> None:
        """起始格算作已访问；重置本身不发放探索奖励。"""
        self.visited.fill(False)
        self.blank_exposure_s.fill(0.0)
        self.current_cell = self.cell_index(position)
        self.visited[self.current_cell] = True
        self.novel_cells = 1
        self.no_new_age_s = 0.0
        self.budget_remaining = self.exploration_budget
        self.exploration_reward_total = 0.0
        self.blank_revisit_cost_total = 0.0
        self.blank_revisit_steps = 0
        self.new_cell = False
        self.blank_revisit_active = False

    def update(
        self,
        position: tuple[float, float],
        *,
        dt: float,
        odor_hit: bool,
        blank_age_s: float,
        terminal: bool,
    ) -> dict[str, float]:
        """在本步采样完成后更新；成功、越界、超时步均免除新奖惩。"""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("空间记忆 dt 必须为有限正数")
        if not math.isfinite(blank_age_s) or blank_age_s < 0.0:
            raise ValueError("blank_age_s 必须为有限非负数")
        self.blank_exposure_s *= math.exp(-dt / self.blank_decay_tau_s)
        self.current_cell = self.cell_index(position)
        cell = self.current_cell
        if odor_hit:
            self.blank_exposure_s[cell] = 0.0
        else:
            self.blank_exposure_s[cell] = min(
                self.blank_cap_s, self.blank_exposure_s[cell] + dt
            )
        self.new_cell = not bool(self.visited[cell])
        if self.new_cell:
            self.visited[cell] = True
            self.novel_cells += 1
            self.no_new_age_s = 0.0
        else:
            self.no_new_age_s = min(self.no_new_age_cap_s, self.no_new_age_s + dt)

        novelty_reward = 0.0
        if self.new_cell and not terminal:
            novelty_reward = min(
                self.budget_remaining,
                self.exploration_bonus * min(blank_age_s / self.blank_reward_delay_s, 1.0),
            )
            self.budget_remaining = max(0.0, self.budget_remaining - novelty_reward)
            self.exploration_reward_total += novelty_reward
        self.blank_revisit_active = bool(
            not terminal
            and not odor_hit
            and blank_age_s + 1e-12 >= self.blank_reward_delay_s
            and self.no_new_age_s + 1e-12 >= self.no_new_age_cap_s
            and self.blank_exposure_s[cell] > self.revisit_start_s
        )
        revisit_cost = 0.0
        if self.blank_revisit_active:
            self.blank_revisit_steps += 1
            revisit_cost = self.blank_revisit_penalty * float(np.clip(
                (self.blank_exposure_s[cell] - self.revisit_start_s) / self.revisit_ramp_s,
                0.0, 1.0,
            ))
            self.blank_revisit_cost_total += revisit_cost
        return {"exploration_bonus": novelty_reward, "blank_revisit_penalty": -revisit_cost}

    def observation(self, *, masked: bool = False) -> np.ndarray:
        """只输出当前地图，历史包装器不得对地图重复堆叠。"""
        if masked:
            return np.zeros(SPATIAL_OBSERVATION_DIM, dtype=np.float32)
        return np.concatenate([
            self.visited.ravel().astype(np.float32),
            (self.blank_exposure_s / self.blank_cap_s).ravel().astype(np.float32),
            np.array([
                self.no_new_age_s / self.no_new_age_cap_s,
                self.budget_remaining / self.budget_normalizer,
            ], dtype=np.float32),
        ])

    @staticmethod
    def field_names() -> list[str]:
        return [f"{name}_y{row}_x{column}" for name in ("visited", "recent_blank")
                for row in range(GRID_SIDE) for column in range(GRID_SIDE)] + [
            "spatial_no_new_age_norm", "exploration_budget_remaining_norm",
        ]

    def info(self) -> dict[str, Any]:
        """轨迹只记录标量诊断，避免每步复制两幅地图。"""
        return {
            "novel_cells": self.novel_cells,
            "new_cell": self.new_cell,
            "spatial_cell_row": self.current_cell[0],
            "spatial_cell_column": self.current_cell[1],
            "spatial_no_new_age_s": self.no_new_age_s,
            "recent_blank_exposure_s": float(self.blank_exposure_s[self.current_cell]),
            "exploration_budget_remaining": self.budget_remaining,
            "exploration_reward_total": self.exploration_reward_total,
            "blank_revisit_active": self.blank_revisit_active,
            "blank_revisit_steps": self.blank_revisit_steps,
            "blank_revisit_cost_total": self.blank_revisit_cost_total,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "episodic_spatial_memory_v1",
            "world_bounds": [-5.0, 5.0],
            "grid_shape": [GRID_SIDE, GRID_SIDE],
            "cell_size_m": GRID_CELL_SIZE_M,
            "array_axes": ["y", "x"],
            "flatten_order": "C",
            "blank_decay_tau_s": self.blank_decay_tau_s,
            "blank_cap_s": self.blank_cap_s,
            "no_new_age_cap_s": self.no_new_age_cap_s,
            "blank_reward_delay_s": self.blank_reward_delay_s,
            "revisit_start_s": self.revisit_start_s,
            "revisit_ramp_s": self.revisit_ramp_s,
            "exploration_bonus": self.exploration_bonus,
            "exploration_budget": self.exploration_budget,
            "budget_normalizer": self.budget_normalizer,
            "blank_revisit_penalty": self.blank_revisit_penalty,
            "inputs": ["robot_position", "thresholded_sensor_odor_hit"],
        }
