"""二维雷达、障碍碰撞和移动环境观测维度的确定性诊断。"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.lidar_model import SimulatedLidar2D
from dual_whisker_rl.envs.lidar_model import validate_obstacles
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from train_whisker_only_ppo import ObservationHistoryWrapper


def _expect_value_error(callback, description: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {description}")


def check_lidar_geometry() -> None:
    single_beam = SimulatedLidar2D(num_beams=1, fov_deg=180.0, max_range=2.0)
    empty = np.empty((0, 4), dtype=np.float32)
    world_hit = single_beam.scan(
        x=0.0,
        y=0.0,
        heading=0.0,
        obstacles=empty,
        world_min=-1.0,
        world_max=1.0,
    )
    np.testing.assert_allclose(world_hit, [1.0], atol=1e-6)

    lidar = SimulatedLidar2D(num_beams=12, fov_deg=180.0, max_range=0.6)
    front_obstacle = np.asarray([[0.4, 0.5, -0.25, 0.25]], dtype=np.float32)
    front_scan = lidar.scan(
        x=0.0,
        y=0.0,
        heading=0.0,
        obstacles=front_obstacle,
        world_min=-1.0,
        world_max=1.0,
    )
    np.testing.assert_allclose(front_scan[5], front_scan[6], atol=1e-6)
    expected_center_pair = 0.4 / math.cos(math.radians(7.5))
    np.testing.assert_allclose(
        [front_scan[5], front_scan[6]],
        [expected_center_pair, expected_center_pair],
        atol=1e-6,
    )

    rotated_obstacle = np.asarray([[-0.25, 0.25, 0.4, 0.5]], dtype=np.float32)
    rotated_scan = lidar.scan(
        x=0.0,
        y=0.0,
        heading=math.pi / 2.0,
        obstacles=rotated_obstacle,
        world_min=-1.0,
        world_max=1.0,
    )
    np.testing.assert_allclose(rotated_scan, front_scan, atol=1e-6)

    clipped = lidar.scan(
        x=0.0,
        y=0.0,
        heading=0.0,
        obstacles=empty,
        world_min=-1.0,
        world_max=1.0,
    )
    np.testing.assert_allclose(clipped, np.full(12, 0.6), atol=1e-6)


def check_obstacle_validation() -> None:
    _expect_value_error(
        lambda: validate_obstacles([[0.0, 0.1, 0.2]], world_min=-1.0, world_max=1.0),
        "wrong obstacle shape",
    )
    _expect_value_error(
        lambda: validate_obstacles([[0.2, 0.1, -0.2, 0.2]], world_min=-1.0, world_max=1.0),
        "negative obstacle width",
    )
    _expect_value_error(
        lambda: validate_obstacles([[-1.1, -0.9, -0.2, 0.2]], world_min=-1.0, world_max=1.0),
        "obstacle outside world",
    )


def check_environment_observations() -> None:
    default_env = MobileWhiskerPuffEnv()
    default_obs, default_info = default_env.reset(seed=7)
    assert default_obs.shape == (16,)
    assert default_env.observation_space.contains(default_obs)
    assert np.asarray(default_info["lidar_ranges_m"]).shape == (12,)

    obstacle_config = {
        "scenario_mode": "fixed",
        "source_position": (-0.8, 0.0),
        "obstacles": [[-0.05, 0.10, -0.45, 0.45], [-0.52, -0.32, 0.32, 0.58]],
        "include_lidar_observation": True,
    }
    obstacle_env = MobileWhiskerPuffEnv(obstacle_config)
    obstacle_obs, _ = obstacle_env.reset(seed=7)
    assert obstacle_obs.shape == (28,)
    assert obstacle_env.observation_space.contains(obstacle_obs)
    assert obstacle_env.plume.is_position_valid(
        obstacle_env.robot_state.x,
        obstacle_env.robot_state.y,
        margin=obstacle_env.robot_radius,
    )
    assert obstacle_env.plume.is_position_valid(
        obstacle_env.plume.source_x,
        obstacle_env.plume.source_y,
        margin=obstacle_env.plume.source_clearance,
    )

    default_history = ObservationHistoryWrapper(default_env, history_length=20)
    obstacle_history = ObservationHistoryWrapper(obstacle_env, history_length=20)
    assert default_history.observation_space.shape == (320,)
    assert obstacle_history.observation_space.shape == (560,)
    default_history.close()
    obstacle_history.close()


def check_collision_termination() -> None:
    env = MobileWhiskerPuffEnv(
        {
            "scenario_mode": "fixed",
            "source_position": (-0.8, 0.0),
            "obstacles": [[0.09, 0.20, -0.20, 0.20]],
            "include_lidar_observation": True,
        }
    )
    env.reset(seed=11)
    env.robot_state = env.robot.reset(0.0, 0.0, 0.0)
    env.prev_distance = env._distance_to_source()
    before = (env.robot_state.x, env.robot_state.y, env.robot_state.heading)
    _, _, terminated, truncated, info = env.step([0, 5, 5])
    after = (env.robot_state.x, env.robot_state.y, env.robot_state.heading)
    assert terminated and not truncated
    assert info["collision"] is True
    assert info["termination_reason"] == "collision"
    assert info["reward_components"]["collision_penalty"] == -25.0
    np.testing.assert_allclose(after, before, atol=1e-12)
    assert float(info["min_lidar_range_m"]) <= env.robot_radius + 0.02
    env.close()


def main() -> None:
    check_lidar_geometry()
    check_obstacle_validation()
    check_environment_observations()
    check_collision_termination()
    print("mobile_lidar_diagnostics=PASS")


if __name__ == "__main__":
    main()
