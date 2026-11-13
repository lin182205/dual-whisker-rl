"""连续底盘与离散触须的扁平动作接口，不依赖训练库。"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from dual_whisker_rl.envs.robot_model import RobotState


class HybridMotionWhiskerSpace(spaces.Box):
    """SB3 使用四维 Box 存储，最后两项的语义严格为分类编号。"""

    VERSION = "hybrid_motion_whisker_v1"

    def __init__(self, sector_count=10, seed=None, categorical_branches=2):
        self.sector_count = int(sector_count)
        self.categorical_branches = int(categorical_branches)
        if self.categorical_branches not in (0, 1, 2):
            raise ValueError("混合动作仅支持0/1/2个触须分类分支")
        if self.sector_count < 2:
            raise ValueError("触须扇区数至少为2")
        super().__init__(low=np.array([-1., -1.]+[0.]*self.categorical_branches, dtype=np.float32),
                         high=np.array([1., 1.]+[self.sector_count-1]*self.categorical_branches, dtype=np.float32),
                         dtype=np.float32, seed=seed)

    def sample(self, mask=None, probability=None):
        if mask is not None or probability is not None:
            raise ValueError("混合动作不支持采样掩码或外部分布")
        return np.concatenate((self.np_random.uniform(-1., 1., 2),
                               self.np_random.integers(self.sector_count, size=getattr(self, 'categorical_branches', 2)))).astype(np.float32)

    def contains(self, x):
        try:
            a = np.asarray(x, dtype=float)
            return bool(a.shape == self.shape and np.isfinite(a).all()
                        and (a >= self.low).all() and (a <= self.high).all()
                        and (a[2:] == np.floor(a[2:])).all())
        except (TypeError, ValueError):
            return False

    def __eq__(self, other):
        return (isinstance(other, HybridMotionWhiskerSpace) and self.sector_count == other.sector_count
                and self.shape == other.shape)


def decode_hybrid_action(action, sector_count=10):
    """只限幅速度；不将连续数值取整，也不将非法扇区悄悄取整。"""
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != 4 or not np.isfinite(a).all():
        raise ValueError("混合动作需要4个有限数值 [linear_norm, angular_norm, left_sector, right_sector]")
    if ((a[2:] != np.floor(a[2:])).any() or (a[2:] < 0).any()
            or (a[2:] >= sector_count).any()):
        raise ValueError("触须动作必须是有效的整数扇区")
    return float(np.clip(a[0], -1., 1.)), float(np.clip(a[1], -1., 1.)), int(a[2]), int(a[3])


def arc_pose(state, linear, angular, duration):
    """精确单轮车圆弧积分，sinc 形式在零角速度附近连续。"""
    turn = angular*duration
    distance = linear*duration*float(np.sinc(turn/(2.*math.pi)))
    mid_heading = state.heading + turn/2.
    return RobotState(state.x+distance*math.cos(mid_heading),
                      state.y+distance*math.sin(mid_heading),
                      (state.heading+turn+math.pi) % (2.*math.pi)-math.pi)


def classify_motion(linear, angular):
    """诊断按实际执行速度归类，不表示策略选择了离散移动动作。"""
    if abs(linear) <= 1e-8:
        return "stop" if abs(angular) <= 1e-8 else ("spin_left" if angular > 0 else "spin_right")
    return "forward" if abs(angular) <= 1e-8 else ("turn_left" if angular > 0 else "turn_right")


class HybridMotion:
    def __init__(self, cfg, robot):
        self.v_max, self.w_max = robot.forward_speed, robot.turn_rate
        if not np.isfinite([self.v_max, self.w_max]).all() or min(self.v_max, self.w_max) <= 0:
            raise ValueError("连续底盘最大线/角速度必须为有限正数")
        self.warning_clearance = float(cfg.setdefault("near_obstacle_clearance", .20))
        self.near_cost = float(cfg.setdefault("near_obstacle_cost", .05))
        self.smooth_cost = float(cfg.setdefault("control_smoothness_cost", .005))
        if (not np.isfinite([self.warning_clearance, self.near_cost, self.smooth_cost]).all()
                or self.warning_clearance <= 0 or min(self.near_cost, self.smooth_cost) < 0):
            raise ValueError("近障距离必须为正，新增成本必须有限非负")
        self.reset()

    def reset(self):
        self.requested_linear = self.requested_angular = 0.
        self.linear = self.angular = 0.
        self.linear_change = self.angular_change = 0.

    def move(self, env, linear_norm, angular_norm):
        self.requested_linear = (linear_norm+1.)*.5*self.v_max
        self.requested_angular = angular_norm*self.w_max
        previous_linear, previous_angular = self.linear, self.angular
        env.robot_state = env.room_navigation.move_arc(
            env.robot_state, self.requested_linear, self.requested_angular, env.dt)
        env.robot.state = env.robot_state
        self.linear = 0. if env.room_navigation.collision else self.requested_linear
        self.angular = self.requested_angular
        self.linear_change = abs(self.linear-previous_linear)
        self.angular_change = abs(self.angular-previous_angular)
        return classify_motion(self.linear, self.angular)

    def info(self, env):
        clearance = max(0., float(np.min(env.room_navigation.scan(env)))-env.room_navigation.radius)
        return {"requested_linear_velocity": self.requested_linear,
                "requested_angular_velocity": self.requested_angular,
                "actual_linear_velocity": self.linear, "actual_angular_velocity": self.angular,
                "linear_velocity_change": self.linear_change, "angular_velocity_change": self.angular_change,
                "control_change_norm": self.linear_change/self.v_max+self.angular_change/(2.*self.w_max),
                "surface_clearance_m": clearance, "near_obstacle": clearance < self.warning_clearance}

    def reward_components(self, env):
        info = self.info(env)
        return {"near_obstacle_penalty": -self.near_cost*float(np.clip(
                    1.-info["surface_clearance_m"]/self.warning_clearance, 0., 1.))**2,
                "control_smoothness_penalty": -self.smooth_cost*info["control_change_norm"]}
