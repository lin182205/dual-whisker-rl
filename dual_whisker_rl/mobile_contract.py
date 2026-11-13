"""独立 Mobile 入口的模型/环境契约和导航指标。"""

from __future__ import annotations

import json

import numpy as np
from dual_whisker_rl.envs.hybrid_action import HybridMotionWhiskerSpace


def environment_contract(env, history_length):
    """将契约写入 PPO 自定义属性，checkpoint 和最终 ZIP 都会携带。"""
    nav = env.room_navigation
    config = {k: v for k, v in env.effective_config.items() if k not in ("seed", "record_trajectory")}
    contract = {
        "observation_version": nav.OBSERVATION_VERSION if nav else "mobile_legacy_v1",
        "field_names": list(env.observation_field_names),
        "history_length": int(history_length),
        "map_sha256": nav.map.digest if nav else None,
        "effective_config": config,
    }
    if env.hybrid_motion:
        motion = env.hybrid_motion
        contract["action_contract"] = {
            "version": HybridMotionWhiskerSpace.VERSION,
            "fields": ["linear_norm", "angular_norm", "left_sector", "right_sector"],
            "continuous_bounds": [[-1., 1.], [-1., 1.]],
            "linear_velocity_m_s": [0., motion.v_max], "angular_velocity_rad_s": [-motion.w_max, motion.w_max],
            "categorical_branches": [env.whiskers.sector_count]*2,
            "distribution": f"Normal(2)+Categorical({env.whiskers.sector_count})+Categorical({env.whiskers.sector_count})",
        }
        contract["radar"] = {"raw_rays": 360, "sectors": 36, "range_m": env.room_navigation.lidar_range,
                             "pooling": "minimum", "sector_zero": "forward_centered",
                             "order": "counterclockwise", "index_rule": "((degrees+5)//10)%36"}
        contract["feature_groups"] = {"olfactory_history": list(range(12))+[16],
                                      "navigation_current": [12, 13, 14, 15]+list(range(17, 57))}
    else:
        contract["action_nvec"] = env.action_space.nvec.tolist()
    return json.loads(json.dumps(contract, sort_keys=True))


def validate_model_environment(model, env, history_length=None, *, resume=False):
    """不能仅用整除推断历史长度：旧 16维模型可能偶然整除新维度。"""
    base_dim = int(env.observation_space.shape[0])
    shape = tuple(model.observation_space.shape)
    if len(shape) != 1 or shape[0] % base_dim:
        raise ValueError(f"模型观测维度 {shape} 与环境单帧维度 {base_dim} 不兼容")
    stored = getattr(model, "mobile_environment_contract", None)
    if env.room_navigation and stored is None:
        raise ValueError("房间环境需要对应观测版本的模型契约；旧模型不能用于新环境")
    inferred = shape[0]//base_dim
    if history_length is not None and int(history_length) != inferred:
        raise ValueError("模型历史长度与本次参数不兼容")
    extractor = model.policy.features_extractor
    if getattr(extractor, "base_observation_dim", base_dim) != base_dim or getattr(extractor, "history_length", inferred) != inferred:
        raise ValueError("模型时序编码器与环境观测定义不兼容")
    if env.hybrid_motion:
        if not isinstance(model.action_space, HybridMotionWhiskerSpace) or model.action_space != env.action_space:
            raise ValueError("模型动作空间与连续/离散混合动作不兼容")
    elif not np.array_equal(getattr(model.action_space, "nvec", None), env.action_space.nvec):
        raise ValueError("模型动作空间与环境不兼容")
    if stored is not None:
        expected = environment_contract(env, inferred)
        # 普通评估允许显式加大/缩小回合上限，物理、观测与地图必须一致；续训严格比较。
        actual = json.loads(json.dumps(stored))
        if env.hybrid_motion:
            from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
            if not isinstance(model.policy, HybridActorCriticPolicy):
                raise ValueError("混合环境必须使用联合高斯/分类策略")
            encoder = getattr(extractor, "configuration", None)
            if encoder is None or actual.get("encoder_config") != encoder:
                raise ValueError("混合模型的编码器配置契约不一致")
            expected["encoder_config"] = encoder
        if not resume:
            expected["effective_config"].pop("max_steps", None)
            actual["effective_config"].pop("max_steps", None)
        if actual != expected:
            raise ValueError("模型与环境契约不一致（地图、字段、历史长度或有效配置发生变化）")
    return inferred


def navigation_episode_metrics(trajectory):
    if not trajectory or "collision" not in trajectory[0]:
        return {}
    first, last = trajectory[0], trajectory[-1]
    collisions = sum(bool(row["collision"]) for row in trajectory)
    length = float(last["path_length"])
    success = bool(last.get("is_success", False))
    initial = float(first["initial_path_distance"])
    # 成功允许停在 goal_radius 内，最短必要路程也扣除该终点余量。
    required = max(0., initial-float(last.get("goal_radius", .1)))
    metrics = {"collision_count": collisions, "collision_rate": collisions/len(trajectory),
            "had_collision": collisions > 0, "path_length": length,
            "initial_path_distance": initial,
            "final_path_distance": float(last["path_distance_to_source"]),
            "path_efficiency": min(1., required/length) if success and length > 0 else None,
            "start_room": first["start_room"], "source_room": first["source_room"],
            "room_relation": "same_room" if first["start_room"] == first["source_room"] else "cross_room"}
    if "actual_linear_velocity" in first:
        linear = np.array([r["actual_linear_velocity"] for r in trajectory])
        angular = np.array([r["actual_angular_velocity"] for r in trajectory])
        metrics.update({"mean_linear_velocity": float(linear.mean()), "max_linear_velocity": float(linear.max()),
                        "mean_abs_angular_velocity": float(np.abs(angular).mean()),
                        "min_surface_clearance_m": min(r["surface_clearance_m"] for r in trajectory),
                        "near_obstacle_step_ratio": float(np.mean([r["near_obstacle"] for r in trajectory])),
                        "mean_control_change_norm": float(np.mean([r["control_change_norm"] for r in trajectory])),
                        "total_linear_velocity_change": sum(r["linear_velocity_change"] for r in trajectory),
                        "total_angular_velocity_change": sum(r["angular_velocity_change"] for r in trajectory)})
        metrics["motion_state_counts"] = {name: sum(r["move_action"] == name for r in trajectory)
                                         for name in ("forward", "turn_left", "turn_right", "spin_left", "spin_right", "stop")}
    return metrics


def aggregate_navigation_metrics(rows):
    if not rows:
        return {"episodes": 0}
    def mean(key):
        v = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(v)) if v else None
    result = {"episodes": len(rows), "success_rate": mean("success"),
              "collision_rate": mean("collision_rate"), "collision_episode_rate": mean("had_collision"),
              "mean_collision_count": mean("collision_count"), "mean_path_efficiency": mean("path_efficiency"),
              "mean_final_path_distance": mean("final_path_distance")}
    if any("mean_linear_velocity" in r for r in rows):
        result.update({key: mean(key) for key in ("mean_linear_velocity", "mean_abs_angular_velocity",
                       "near_obstacle_step_ratio", "mean_control_change_norm")})
        result["max_linear_velocity"] = max(r["max_linear_velocity"] for r in rows)
        for key in ("total_linear_velocity_change", "total_angular_velocity_change"):
            result[key] = sum(r[key] for r in rows)
        result["min_surface_clearance_m"] = min(r["min_surface_clearance_m"] for r in rows)
        result["motion_state_counts"] = {name: sum(r["motion_state_counts"].get(name, 0) for r in rows)
                                         for name in rows[0]["motion_state_counts"]}
    return result


def navigation_summary(rows):
    valid = [row for row in rows if "collision_count" in row]
    if not valid:
        return {}
    return {**aggregate_navigation_metrics(valid),
            "by_room_relation": {key: aggregate_navigation_metrics([r for r in valid if r["room_relation"] == key])
                                 for key in ("same_room", "cross_room")}}
