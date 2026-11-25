"""Mobile 双分支的观测字段与可序列化特征定义。"""

WIND_MOTION_HISTORY = "wind_motion_history_v1"
WIND_MOTION_SPATIAL_MEMORY = "wind_motion_spatial_memory_v2"
OLFACTORY_INDICES = tuple(range(12)) + (16,)
HEADING_INDICES = (12, 13)
POSITION_INDICES = (17, 18)
WIND_INDICES = (19, 20)
LIDAR_INDICES = tuple(range(21, 57))
NAVIGATION_INDICES = (12, 13, 14, 15) + POSITION_INDICES + WIND_INDICES + LIDAR_INDICES
CURRENT_NAVIGATION_INDICES = (12, 13, 14, 15) + POSITION_INDICES + LIDAR_INDICES
HISTORY_ABLATIONS = ("none", "current_wind_only", "no_motion_history", "no_absolute_xy")


def feature_layout_contract(feature_layout=None, history_ablation="none"):
    """旧模型保持原契约，新的编码方式独立于原始57维字段版本。"""
    if feature_layout is None and history_ablation == "none":
        return None
    if feature_layout not in (WIND_MOTION_HISTORY, WIND_MOTION_SPATIAL_MEMORY):
        raise ValueError("未知风向与运动历史特征布局")
    if history_ablation not in HISTORY_ABLATIONS:
        raise ValueError(f"未知历史消融：{history_ablation!r}")
    contract = {
        "version": feature_layout,
        "history_ablation": history_ablation,
        "history_width": 19,
        "navigation_width": 42,
        "history_fields_after_olfactory": [
            "wind_cos_current_body", "wind_sin_current_body", "past_forward_m", "past_left_m",
            "past_relative_heading_cos", "past_relative_heading_sin",
        ],
        "heading_indices": list(HEADING_INDICES),
        "position_indices": list(POSITION_INDICES),
        "relative_wind_indices": list(WIND_INDICES),
        "position_m_per_normalized_unit": 5.0,
        "reference_frame": "latest_body_pose",
    }
    if feature_layout == WIND_MOTION_SPATIAL_MEMORY:
        contract["spatial_memory"] = {
            "core_width": 57, "grid_shape": [20, 20], "channels": ["visited", "recent_blank_seconds/5"],
            "bounds_m": [-5.0, 5.0], "cell_size_m": 0.5, "flatten_order": "channel_y_x",
            "scalar_fields": ["no_new_cell_age_seconds/5", "exploration_budget_remaining/0.5"],
            "history_packing": "core_history_then_current_memory", "tail_width": 802,
            "cnn_position_planes": "cell_center_minus_robot_xy_divided_by_10",
        }
    return contract


def feature_layout_kwargs(config):
    """独立和批量入口使用同一套配置字段构造编码器。"""
    return {key: config[key] for key in ("feature_layout", "history_ablation") if key in config}
