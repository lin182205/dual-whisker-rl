"""E4 仿真对比实验的可复用场景、方法适配和评估工具。

该模块刻意不依赖命令行状态：场景文件、方法定义和评估函数都可以被测试或
其它脚本直接导入。所有固定场景通过 MobileWhiskerPuffEnv.reset(options=...)
注入，避免为比较实验复制一套物理环境。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.world_bounds import resolve_world_bounds


METHOD_ORDER = (
    "main",
    "b0_random",
    "b1_reactive",
    "b2_fixed",
    "b3_scan",
    "b4_single_frame",
    "b5_history_mlp",
    "b6_left",
    "b6_right",
)
TRAINABLE_METHODS = frozenset({
    "main", "b2_fixed", "b3_scan", "b4_single_frame", "b5_history_mlp",
    "b6_left", "b6_right",
})
RULE_METHODS = frozenset({"b0_random", "b1_reactive"})


@dataclass(frozen=True)
class E4Scenario:
    """可序列化的单个配对测试场景。"""

    scenario_id: str
    split: str
    seed: int
    wind_speed: float
    wind_direction: float
    source_position: tuple[float, float]
    robot_pose: tuple[float, float, float]
    source_strength: float
    distance_bin: str
    crosswind_bin: str
    heading_quadrant: int
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_position"] = list(self.source_position)
        data["robot_pose"] = list(self.robot_pose)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "E4Scenario":
        return cls(
            scenario_id=str(data["scenario_id"]),
            split=str(data["split"]),
            seed=int(data["seed"]),
            wind_speed=float(data["wind_speed"]),
            wind_direction=float(data["wind_direction"]),
            source_position=tuple(float(x) for x in data["source_position"]),
            robot_pose=tuple(float(x) for x in data["robot_pose"]),
            source_strength=float(data["source_strength"]),
            distance_bin=str(data["distance_bin"]),
            crosswind_bin=str(data["crosswind_bin"]),
            heading_quadrant=int(data["heading_quadrant"]),
            repeat=int(data["repeat"]),
        )


class ObservationHistory(gym.Wrapper):
    """轻量历史堆叠包装器，保持与训练入口的观测顺序一致。"""

    def __init__(self, env: gym.Env, history_length: int) -> None:
        super().__init__(env)
        if int(history_length) < 1:
            raise ValueError("history_length must be at least 1")
        self.history_length = int(history_length)
        self.history: deque[np.ndarray] = deque(maxlen=self.history_length)
        if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
            raise TypeError("ObservationHistory requires a flat Box observation space")
        self.base_observation_dim = int(env.observation_space.shape[0])
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low, self.history_length).astype(np.float32),
            high=np.tile(env.observation_space.high, self.history_length).astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(np.asarray(obs, dtype=np.float32).copy())
        return np.concatenate(tuple(self.history)).astype(np.float32), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(np.asarray(obs, dtype=np.float32))
        return np.concatenate(tuple(self.history)).astype(np.float32), reward, terminated, truncated, info


class MethodActionAdapter(gym.Wrapper):
    """把不同基线的动作接口映射到共同的三元底层动作。"""

    def __init__(self, env: gym.Env, method: str, fixed_sector: int = 5, scan_hold: int = 2) -> None:
        super().__init__(env)
        self.method = method
        self.fixed_sector = int(fixed_sector)
        self.scan_hold = max(1, int(scan_hold))
        self._scan_sector = 0
        self._scan_direction = 1
        self._scan_count = 0
        if method == "b2_fixed" or method == "b3_scan":
            self.action_space = spaces.Discrete(6)
        elif method in ("b6_left", "b6_right"):
            self.action_space = spaces.MultiDiscrete([6, 10])
        else:
            self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        self._scan_sector, self._scan_direction, self._scan_count = 0, 1, 0
        obs, info = self.env.reset(seed=seed, options=options)
        if self.method == "b2_fixed":
            # 固定触须从 reset 返回的首帧就落在候选镜像扇区；不消耗一个控制周期。
            self.env.whiskers.left_angle = self.env.whiskers._sector_angle(self.fixed_sector, side="left")
            self.env.whiskers.right_angle = self.env.whiskers._sector_angle(self.fixed_sector, side="right")
            self.env.last_left_sector = self.fixed_sector
            self.env.last_right_sector = self.fixed_sector
            obs = self.env._build_observation()
        return obs, info

    def _map_action(self, action: Any) -> np.ndarray:
        if self.method == "b2_fixed":
            move = int(np.asarray(action).reshape(-1)[0])
            return np.asarray([move, self.fixed_sector, self.fixed_sector], dtype=np.int64)
        if self.method == "b3_scan":
            move = int(np.asarray(action).reshape(-1)[0])
            sector = self._scan_sector
            self._scan_count += 1
            if self._scan_count >= self.scan_hold:
                self._scan_count = 0
                self._scan_sector += self._scan_direction
                if self._scan_sector >= 9:
                    self._scan_sector, self._scan_direction = 9, -1
                elif self._scan_sector <= 0:
                    self._scan_sector, self._scan_direction = 0, 1
            return np.asarray([move, sector, sector], dtype=np.int64)
        if self.method in ("b6_left", "b6_right"):
            arr = np.asarray(action, dtype=np.int64).reshape(-1)
            move, sector = int(arr[0]), int(arr[1])
            if self.method == "b6_left":
                return np.asarray([move, sector, self.fixed_sector], dtype=np.int64)
            return np.asarray([move, self.fixed_sector, sector], dtype=np.int64)
        return np.asarray(action, dtype=np.int64)

    def step(self, action: Any):
        return self.env.step(self._map_action(action))


def choose_mlp_width(input_dim: int, *, target_parameters: int = 60000, output_dim: int = 320) -> int:
    """按输入维度估算两层等宽 MLP 宽度，尽量接近指定参数预算。

    这是可解释的架构匹配启发式，不把参数量写死在实验结果中；实际参数量
    会随 SB3 的动作头和价值头变化，并在 metadata 中记录。
    """
    input_dim = max(1, int(input_dim))
    target_parameters = max(1000, int(target_parameters))
    # 输入层、隐藏层和一个共享输出头的主要项：d*w + w*w + w*output。
    a = 1.0
    b = float(input_dim + output_dim)
    c = -float(target_parameters)
    width = int(max(8.0, (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)))
    return min(max(width, 16), 256)


def method_metadata(method: str) -> dict[str, Any]:
    if method not in METHOD_ORDER:
        raise ValueError(f"unknown E4 method: {method}")
    return {
        "method": method,
        "label": {
            "main": "M: GRU joint",
            "b0_random": "B0: random",
            "b1_reactive": "B1: reactive",
            "b2_fixed": "B2: fixed whisker PPO",
            "b3_scan": "B3: periodic scan PPO",
            "b4_single_frame": "B4: single-frame MLP PPO",
            "b5_history_mlp": "B5: history MLP PPO",
            "b6_left": "B6-left: single whisker PPO",
            "b6_right": "B6-right: single whisker PPO",
        }[method],
        "trainable": method in TRAINABLE_METHODS,
        "action_interface": "MultiDiscrete[6,10,10]",
        "history_length": 1 if method == "b4_single_frame" else 20,
        "temporal_encoder": "gru" if method in {"main", "b2_fixed", "b3_scan", "b6_left", "b6_right"} else "mlp",
        "active_whisker_side": "left" if method == "b6_left" else "right" if method == "b6_right" else None,
    }


def _valid_pose(
    source: np.ndarray,
    wind: np.ndarray,
    downwind: float,
    crosswind: float,
    world_bounds: tuple[float, float],
    margin: float = 0.16,
) -> tuple[float, float] | None:
    cross = np.asarray([-wind[1], wind[0]], dtype=np.float64)
    robot = source + downwind * wind + crosswind * cross
    world_min, world_max = (float(world_bounds[0]), float(world_bounds[1]))
    if (
        world_min + margin < robot[0] < world_max - margin
        and world_min + margin < robot[1] < world_max - margin
    ):
        return float(robot[0]), float(robot[1])
    return None


def build_scenarios(
    split: str = "test",
    profile: str = "formal",
    seed: int = 20260915,
    world_bounds: tuple[float, float] | list[float] | None = None,
) -> list[E4Scenario]:
    """生成固定分层场景；几何尺度随 world_bounds 缩放。"""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    world_min, world_max, world_half = resolve_world_bounds(world_bounds, default_half=2.5)
    bounds = (world_min, world_max)
    rng = np.random.default_rng(seed + (11 if split == "validation" else 29))
    wind_speeds = (0.12, 0.16, 0.20)
    distances = (("near", 0.75 * world_half), ("middle", 1.00 * world_half), ("far", 1.25 * world_half))
    crosswinds = (("center", 0.05 * world_half), ("edge", 0.30 * world_half))
    strengths = (0.7, 1.0, 1.3)
    quadrants = (0, 1, 2, 3)
    repeats = (0,) if split == "validation" else (0, 1, 2)
    scenarios: list[E4Scenario] = []
    for wi, speed in enumerate(wind_speeds):
        for di, (distance_bin, distance) in enumerate(distances):
            for ci, (crosswind_bin, crosswind_abs) in enumerate(crosswinds):
                for si, strength in enumerate(strengths):
                    for quadrant in quadrants:
                        for repeat in repeats:
                            # 源位于上风侧约 0.8 个场地半宽；几何参数按新场地缩放。
                            cross_sign = -1.0 if (repeat + ci + si) % 2 else 1.0
                            pose_xy = None
                            source = None
                            wind_angle = 0.0
                            relative_heading = 0.0
                            for _ in range(256):
                                wind_angle = float(rng.uniform(-math.pi, math.pi))
                                relative_heading = quadrant * math.pi / 2.0 + float(rng.uniform(-0.12, 0.12))
                                wind = np.asarray([math.cos(wind_angle), math.sin(wind_angle)], dtype=np.float64)
                                cross = np.asarray([-wind[1], wind[0]], dtype=np.float64)
                                source_candidate = -0.80 * world_half * wind + cross_sign * crosswind_abs * cross
                                if not (
                                    world_min + 0.08 < source_candidate[0] < world_max - 0.08
                                    and world_min + 0.08 < source_candidate[1] < world_max - 0.08
                                ):
                                    continue
                                candidate_pose = _valid_pose(
                                    source_candidate, wind, distance, cross_sign * crosswind_abs, bounds
                                )
                                if candidate_pose is not None:
                                    source = source_candidate
                                    pose_xy = candidate_pose
                                    break
                            if pose_xy is None or source is None:
                                raise RuntimeError(f"cannot place scenario {wi}/{di}/{ci}/{si}/{quadrant}/{repeat} inside world_bounds={bounds}")
                            heading = float((wind_angle + math.pi + relative_heading + math.pi) % (2 * math.pi) - math.pi)
                            sid = f"{split}_w{wi}_d{di}_c{ci}_s{si}_q{quadrant}_r{repeat}"
                            scenarios.append(E4Scenario(
                                scenario_id=sid,
                                split=split,
                                seed=int(seed + len(scenarios) * 17 + 1000 * (split == "test")),
                                wind_speed=float(speed),
                                wind_direction=float(wind_angle),
                                source_position=(float(source[0]), float(source[1])),
                                robot_pose=(pose_xy[0], pose_xy[1], heading),
                                source_strength=float(strength),
                                distance_bin=distance_bin,
                                crosswind_bin=crosswind_bin,
                                heading_quadrant=quadrant,
                                repeat=repeat,
                            ))
    if profile == "smoke":
        return scenarios[:12]
    if profile != "formal":
        raise ValueError("profile must be smoke or formal")
    return scenarios


def scenario_hash(scenarios: list[E4Scenario]) -> str:
    payload = json.dumps([s.to_dict() for s in scenarios], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def make_env(method: str, config: dict[str, Any] | None = None, *, history_length: int | None = None, fixed_sector: int = 5) -> gym.Env:
    cfg = dict(config or {})
    meta = method_metadata(method)
    if "world_bounds" not in cfg and "world_half" not in cfg:
        cfg["world_bounds"] = [-2.5, 2.5]
    cfg.setdefault("scenario_mode", "randomized")
    if meta["active_whisker_side"] is not None:
        cfg["active_whisker_side"] = meta["active_whisker_side"]
    env: gym.Env = MobileWhiskerPuffEnv(cfg)
    if method in ("b2_fixed", "b3_scan", "b6_left", "b6_right"):
        env = MethodActionAdapter(env, method, fixed_sector=fixed_sector, scan_hold=2)
    length = int(history_length if history_length is not None else meta["history_length"])
    return ObservationHistory(env, length)


def reset_scenario(env: gym.Env, scenario: E4Scenario):
    return env.reset(seed=scenario.seed, options={"scenario": scenario.to_dict()})


class RandomController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return np.asarray([self.rng.integers(0, 6), self.rng.integers(0, 10), self.rng.integers(0, 10)], dtype=np.int64), None


class ReactiveController:
    """只使用硬件可得的平滑信号、趋势和左右差分的确定性规则。"""
    def __init__(
        self,
        seed: int = 0,
        *, hit_threshold: float = 0.08,
        trend_threshold: float = -0.02,
        contrast_threshold: float = 0.03,
        search_period: int = 8,
        scan_period: int = 2,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.hit_threshold = float(hit_threshold)
        self.trend_threshold = float(trend_threshold)
        self.contrast_threshold = float(contrast_threshold)
        self.search_period = max(1, int(search_period))
        self.scan_period = max(1, int(scan_period))
        self.last_search_direction = 1
        self.steps_since_hit = 0

    def reset(self) -> None:
        self.last_search_direction = 1
        self.steps_since_hit = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)[-16:]
        left_smooth, right_smooth = float(frame[2]), float(frame[3])
        left_trend, right_trend = float(frame[4]), float(frame[5])
        diff = float(frame[6])
        hit = max(left_smooth, right_smooth) > self.hit_threshold
        if hit:
            self.steps_since_hit = 0
            move = 0 if (left_trend + right_trend) >= self.trend_threshold else 5
        else:
            self.steps_since_hit += 1
            if self.steps_since_hit % self.search_period == 0:
                self.last_search_direction *= -1
            move = 3 if self.last_search_direction > 0 else 4
        if abs(diff) > self.contrast_threshold:
            sector = 7 if diff < 0 else 2
            other = 2 if diff < 0 else 7
        else:
            sector = int((self.steps_since_hit // self.scan_period) % 10)
            other = 9 - sector
        return np.asarray([move, sector, other], dtype=np.int64), None


def metrics_from_trajectory(
    trajectory: list[dict[str, Any]],
    goal_radius: float,
    world_bounds: tuple[float, float] = (-2.5, 2.5),
) -> dict[str, Any]:
    if not trajectory:
        return {"success": False, "termination": "empty", "steps": 0, "final_distance": None, "path_length": 0.0, "odor_hits": 0, "odor_hit_ratio": None, "blank_steps": 0, "reacquisition_events": 0, "reacquisition_per_1000_steps": None, "mean_reacquisition_time_s": None}
    path_length = 0.0
    for prev, cur in zip(trajectory, trajectory[1:]):
        path_length += math.hypot(float(cur["x"]) - float(prev["x"]), float(cur["y"]) - float(prev["y"]))
    hits = sum(bool(row.get("odor_hit", False)) for row in trajectory)
    blanks = len(trajectory) - hits
    final_distance = float(trajectory[-1]["distance_to_source"])
    success = bool(final_distance <= goal_radius)
    world_min, world_max = (float(world_bounds[0]), float(world_bounds[1]))
    out_of_bounds = not (
        world_min < float(trajectory[-1]["x"]) < world_max
        and world_min < float(trajectory[-1]["y"]) < world_max
    )
    reacquisition_times = [float(row.get("reacquisition_blank_s", 0.0)) for row in trajectory if row.get("reacquisition_event", False)]
    return {
        "success": success,
        "termination": "success" if success else "out_of_bounds" if out_of_bounds else "timeout",
        "steps": len(trajectory),
        "final_distance": final_distance,
        "path_length": path_length,
        "odor_hits": int(hits),
        "odor_hit_ratio": hits / max(len(trajectory), 1),
        "blank_steps": int(blanks),
        "blank_body_spin_ratio": sum(row.get("move_action") in ("spin_left", "spin_right") for row in trajectory if not row.get("odor_hit", False)) / max(blanks, 1),
        "blank_body_motion_ratio": sum(row.get("body_moved", False) for row in trajectory if not row.get("odor_hit", False)) / max(blanks, 1),
        "blank_whisker_motion_ratio": sum(row.get("whisker_moved", False) for row in trajectory if not row.get("odor_hit", False)) / max(blanks, 1),
        "reacquisition_events": sum(bool(row.get("reacquisition_event", False)) for row in trajectory),
        "reacquisition_per_1000_steps": 1000.0 * sum(bool(row.get("reacquisition_event", False)) for row in trajectory) / max(len(trajectory), 1),
        "mean_reacquisition_time_s": float(np.mean(reacquisition_times)) if reacquisition_times else None,
        "whisker_only_reacquisition_events": sum(bool(row.get("whisker_only_reacquisition", False)) for row in trajectory),
        "body_assisted_reacquisition_events": sum(bool(row.get("body_assisted_reacquisition", False)) for row in trajectory),
        "passive_reacquisition_events": sum(bool(row.get("passive_reacquisition", False)) for row in trajectory),
    }


def evaluate_one(model: Any, env: gym.Env, scenario: E4Scenario, *, deterministic: bool = True, max_steps: int | None = None) -> dict[str, Any]:
    obs, info = reset_scenario(env, scenario)
    if hasattr(model, "reset"):
        model.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    base = env.unwrapped
    row = metrics_from_trajectory(
        list(base.trajectory),
        float(base.goal_radius),
        (float(base.world_min), float(base.world_max)),
    )
    if getattr(base, "active_whisker_side", None) is None:
        row["mean_left_right_contrast"] = float(np.mean([abs(float(x.get("left", 0.0)) - float(x.get("right", 0.0))) for x in base.trajectory])) if base.trajectory else 0.0
    else:
        row["mean_left_right_contrast"] = None
    row.update({"scenario_id": scenario.scenario_id, "split": scenario.split, "seed": scenario.seed, "wind_speed": scenario.wind_speed, "distance_bin": scenario.distance_bin, "crosswind_bin": scenario.crosswind_bin, "source_strength": scenario.source_strength, "heading_quadrant": scenario.heading_quadrant, "repeat": scenario.repeat, "reward": float(sum(float(x.get("reward", 0.0)) for x in base.trajectory))})
    return row


def aggregate(rows: list[dict[str, Any]], *, include_breakdown: bool = True) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0, "coverage": 0.0}
    numeric = ("success", "steps", "final_distance", "path_length", "odor_hit_ratio", "reacquisition_events", "whisker_only_reacquisition_events", "blank_body_spin_ratio", "blank_body_motion_ratio", "blank_whisker_motion_ratio", "reacquisition_per_1000_steps", "mean_reacquisition_time_s", "mean_left_right_contrast", "reward")
    successes = [float(r["success"]) for r in rows]
    success_rate = float(np.mean(successes))
    success_std = float(np.std(successes, ddof=1)) if len(successes) > 1 else 0.0
    success_margin = 1.96 * success_std / math.sqrt(len(successes)) if len(successes) > 1 else 0.0
    out: dict[str, Any] = {"episodes": len(rows), "coverage": 1.0, "success_rate": success_rate, "success_std": success_std, "success_ci95": [success_rate - success_margin, success_rate + success_margin]}
    for name in numeric[1:]:
        vals = [float(r[name]) for r in rows if r.get(name) is not None]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[f"mean_{name}"] = mean
            out[f"std_{name}"] = std
            margin = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            out[f"ci95_{name}"] = [mean - margin, mean + margin]
    if include_breakdown:
        out["by_distance_bin"] = {key: aggregate([r for r in rows if r["distance_bin"] == key], include_breakdown=False) for key in sorted({r["distance_bin"] for r in rows})}
        out["by_wind_speed"] = {str(key): aggregate([r for r in rows if r["wind_speed"] == key], include_breakdown=False) for key in sorted({r["wind_speed"] for r in rows})}
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
