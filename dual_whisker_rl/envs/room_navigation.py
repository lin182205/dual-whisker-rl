"""Mobile 房间模式的导航状态和新增观测，不改变旧场景默认语义。"""

from __future__ import annotations

import math

import numpy as np

from dual_whisker_rl.envs.robot_model import RobotState
from dual_whisker_rl.envs.hybrid_action import arc_pose
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.envs.whisker_model import DualWhiskerSampler, WhiskerState


class WallAwareWhiskers(DualWhiskerSampler):
    def __init__(self, original, room_map):
        self.__dict__.update(original.__dict__)
        self.room_map = room_map

    def state(self, robot_state):
        s = super().state(robot_state)
        origin = (robot_state.x, robot_state.y)
        return WhiskerState(s.left_angle, s.right_angle,
                            self.room_map.clip_sample(origin, s.left_point),
                            self.room_map.clip_sample(origin, s.right_point))


class RoomNavigation:
    OBSERVATION_VERSION = "mobile_rooms_v1"

    def __init__(self, cfg):
        if cfg["map_layout"] not in RoomMap.LAYOUTS:
            raise ValueError("未知房间地图版本")
        if list(cfg.get("world_bounds", [-5., 5.])) != [-5., 5.]:
            raise ValueError("固定房间布局需要 world_bounds=[-5,5]")
        if cfg.get("plume_model", "dynamic") != "dynamic":
            raise ValueError("房间模式仅支持 dynamic 羽流")
        if not cfg.get("include_blank_age_observation", True):
            raise ValueError("mobile_rooms_v1 必须保留原16维含 blank_age 观测")
        if "obstacles" in cfg or "obstacles" in (cfg.get("plume_overrides") or {}):
            raise ValueError("固定房间布局的障碍物仅由房间墙体定义")
        self.map = RoomMap(layout=cfg["map_layout"])
        self.radius = float(cfg.get("robot_radius", .15))
        self.source_clearance = float(cfg.get("source_clearance", .25))
        self.minimum_distance = float(cfg.get("minimum_start_path_distance", 1.))
        self.beams = int(cfg.get("lidar_beams", 36))
        self.hybrid = cfg.get("motion_mode") == "hybrid"
        if self.hybrid:
            self.OBSERVATION_VERSION = "mobile_hybrid_nav_v1"
            if self.beams != 36 or cfg.get("lidar_raw_beams", 360) != 360:
                raise ValueError("mobile_hybrid_nav_v1 固定为360原始射线与36扇区")
        if float(cfg.get("lidar_beams", 36)) != self.beams:
            raise ValueError("lidar_beams 必须是整数")
        self.lidar_range = float(cfg.get("lidar_range", 3.))
        self.collision_penalty = float(cfg.get("collision_penalty", .5))
        for value in (self.radius, self.source_clearance, self.minimum_distance, self.lidar_range):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("房间几何参数必须是有限正数")
        if self.radius >= .5 or self.source_clearance < self.radius or self.beams < 1:
            raise ValueError("安全半径必须小于半个门宽，源间距不能小于半径，雷达束数必须为正")
        if not math.isfinite(self.collision_penalty) or self.collision_penalty < 0:
            raise ValueError("collision_penalty 必须是有限非负数")
        cfg.setdefault("world_bounds", [-5., 5.])
        cfg.setdefault("max_steps", 1200)
        cfg.setdefault("oob_penalty", 60.)
        overrides = cfg.setdefault("plume_overrides", {})
        overrides.setdefault("puff_warmup_steps", 500)
        overrides.setdefault("max_puffs", 1200)
        self.field_names = ["position_x_norm", "position_y_norm", "wind_relative_cos", "wind_relative_sin"]
        self.field_names += [f"lidar_{i:02d}_norm" for i in range(self.beams)]
        self.reset()

    def reset(self):
        self.collision = False
        self.collision_count = 0
        self.path_length = 0.
        self.source = None
        self.source_distances = None
        self.start_room = None
        self.initial_path_distance = None
        self._scan_key = None

    def initialize(self, env):
        self.source = (env.plume.source_x, env.plume.source_y)
        self.source_distances = self.map.source_distances(self.source, self.radius)
        if env._scenario_robot_override is not None:
            x, y, heading = env._scenario_robot_override
            if not np.isfinite([x, y, heading]).all() or not self.map.valid((x, y), self.radius):
                raise ValueError("固定场景的机器人位姿位于墙内或超出安全边界")
            distance = self.distance((x, y))
            if not math.isfinite(distance) or distance < self.minimum_distance:
                raise ValueError("固定场景的起点不可达或距源过近")
            mode = "scenario"
        else:
            for _ in range(10000):
                x, y = self.map.sample_free(env._scenario_rng, self.radius)
                distance = self.distance((x, y))
                if math.isfinite(distance) and distance >= self.minimum_distance:
                    break
            else:
                raise RuntimeError("无法采样与气源可达且足够远的机器人起点")
            heading = float(env._scenario_rng.uniform(-np.pi, np.pi))
            mode = "uniform"
        self.start_room = self.map.room_id((x, y))
        self.initial_path_distance = distance
        env.initial_heading_mode = mode
        return env.robot.reset(x, y, heading)

    def distance(self, position):
        return self.map.path_distance(position, self.source, self.radius, self.source_distances)

    @staticmethod
    def potential(distance):
        return -8.*float(np.clip(distance/20., 0., 1.))

    def move(self, previous, proposed):
        a, b = (previous.x, previous.y), (proposed.x, proposed.y)
        self.collision = not bool(self.map.visible(a, b, self.radius))
        if self.collision:
            self.collision_count += 1
            return RobotState(previous.x, previous.y, proposed.heading)
        self.path_length += math.dist(a, b)
        return proposed

    def observation(self, env):
        p = (env.robot_state.x, env.robot_state.y)
        self.lidar_ranges = self.scan(env)
        relative = env.plume.wind_direction-env.robot_state.heading
        return np.concatenate(([p[0]/5., p[1]/5., math.cos(relative), math.sin(relative)],
                               self.lidar_ranges/self.lidar_range)).astype(np.float32)

    def scan(self, env):
        state = env.robot_state
        if self.hybrid:
            key = (state.x, state.y, state.heading)
            if self._scan_key != key:
                self.raw_lidar_ranges = self.map.raycast(key[:2], key[2], 360, self.lidar_range)
                # 原始0°射线属于正前方居中的扇区0，355°至4°循环并入此扇区。
                bins = ((np.arange(360)+5)//10) % 36
                self._scan_ranges = np.full(36, self.lidar_range, dtype=np.float32)
                np.minimum.at(self._scan_ranges, bins, self.raw_lidar_ranges)
                self._scan_key = key
            return self._scan_ranges.copy()
        return self.map.raycast((state.x, state.y), state.heading, self.beams, self.lidar_range)

    def move_arc(self, previous, linear, angular, dt):
        """细分精确圆弧；弦线再按弧弦最大偏离的上界膨胀，防止切角穿墙。"""
        n = max(1, int(math.ceil(abs(linear*dt)/.01)), int(math.ceil(abs(angular*dt)/.05)))
        sub_dt = dt/n
        sagitta_bound = abs(linear*sub_dt)*abs(angular*sub_dt)/8.
        pose = previous
        self.collision = False
        for _ in range(n):
            proposed = arc_pose(pose, linear, angular, sub_dt)
            margin = self.radius+sagitta_bound
            if (not self.map.valid((pose.x, pose.y), margin)
                    or not self.map.valid((proposed.x, proposed.y), margin)
                    or not self.map.visible((pose.x, pose.y), (proposed.x, proposed.y), margin)):
                self.collision = True
                break
            pose = proposed
        final = arc_pose(previous, linear, angular, dt)
        if self.collision:
            self.collision_count += 1
            return RobotState(previous.x, previous.y, final.heading)
        self.path_length += abs(linear*dt)
        return final

    def info(self, env):
        info = {"collision": self.collision, "collision_count": self.collision_count,
                "path_distance_to_source": self.distance((env.robot_state.x, env.robot_state.y)),
                "initial_path_distance": self.initial_path_distance, "path_length": self.path_length,
                "lidar_ranges_m": self.scan(env).tolist(), "start_room": self.start_room,
                "source_room": self.map.room_id(self.source), "map_sha256": self.map.digest}
        if self.hybrid:
            info["raw_lidar_ranges_m"] = self.raw_lidar_ranges.tolist()
        return info
