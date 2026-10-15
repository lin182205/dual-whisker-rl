"""红字消融和故障实验的显式环境适配，不改变原有环境默认行为。"""
from __future__ import annotations
from collections import deque
import math
import gymnasium as gym
import numpy as np
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.whisker_only_env import DynamicPuffPlume
from dual_whisker_rl.e4_experiments import ObservationHistory, MethodActionAdapter

FEATURE_MASKS = {
    "none": [],
    "no_trend": [4,5,7],
    "no_difference": [6,7],
    "no_blank_age": [15],
    "no_angles": [8,9],
    "raw_only": [2,3,4,5,6,7],
}
FAULT_KEYS = {
    "start_s", "wind_speed", "wind_turn_deg", "source_factor", "release_duty",
    "release_period_s", "sensor_tau_factor", "sensor_noise_factor",
    "left_gain", "right_gain", "left_drift_per_s", "right_drift_per_s",
    "sensor_saturation", "sample_drop_probability", "servo_time_factor",
    "action_delay_steps", "command_drop_probability", "heading_noise_deg",
}

class ExperimentPuff(DynamicPuffPlume):
    """域随机化幅度围绕现有标称参数缩放，暖场之前生效。"""
    spread = 1.0
    fault = None
    episode_time_s = 0.0

    def _randomize_params(self):
        super()._randomize_params()
        if self.spread == 1.0:
            return
        for key, nominal in self._dr_nominal.items():
            value = float(nominal) + self.spread*(float(getattr(self,key))-float(nominal))
            if key == "puff_release_per_step":
                value = max(1,int(round(value)))
            else:
                value = max(1e-8,value)
            setattr(self,key,value)

    def _emit_puffs(self):
        fault = self.fault or {}
        duty = float(fault.get("release_duty",1.0))
        period = float(fault.get("release_period_s",8.0))
        if self.episode_time_s >= float(fault.get("start_s",0.0)):
            if self.episode_time_s % period >= duty*period:
                return
        super()._emit_puffs()

class FaultSensor:
    """始终推进物理传感器，丢样只保持读数；故障随机流独立于羽流。"""
    def __init__(self, sensor, fault, side, dt, seed):
        self.sensor,self.fault,self.side,self.dt = sensor,fault,side,float(dt)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.value = float(sensor.value)

    def reset(self,value=0.0):
        self.sensor.reset(value)
        self.steps,self.value = 0,float(self.sensor.value)

    def update(self, concentration):
        reading = float(self.sensor.update(concentration))
        age = self.steps*self.dt
        self.steps += 1
        if age < float(self.fault.get("start_s",0.0)):
            self.value = reading
            return self.value
        gain = float(self.fault.get(f"{self.side}_gain",1.0))
        drift = float(self.fault.get(f"{self.side}_drift_per_s",0.0))
        reading = reading*gain + drift*(age-float(self.fault.get("start_s",0.0)))
        saturation = self.fault.get("sensor_saturation")
        if saturation is not None:
            reading = min(reading,float(saturation))
        if self.rng.random() >= float(self.fault.get("sample_drop_probability",0.0)):
            self.value = reading
        return self.value

class ExperimentMobileEnv(MobileWhiskerPuffEnv):
    """仅实验入口使用的继承环境：特征消融、持续到源和可控故障。"""
    def __init__(self, config):
        cfg = dict(config)
        self.fixed_initial_sector = cfg.pop("fixed_initial_sector", None)
        self.feature_mask = str(cfg.pop("feature_mask","none"))
        self.success_hold_s = float(cfg.pop("success_hold_s",1.0))
        self.dr_spread = float(cfg.pop("dr_spread",1.0))
        self.fault = dict(cfg.pop("fault",{}) or {})
        if self.feature_mask not in FEATURE_MASKS:
            raise ValueError(f"未知特征消融：{self.feature_mask}")
        if set(self.fault)-FAULT_KEYS:
            raise ValueError(f"未知故障字段：{set(self.fault)-FAULT_KEYS}")
        for key,value in self.fault.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"故障参数必须有限：{key}")
        for key in ("sample_drop_probability","command_drop_probability","release_duty"):
            if not 0 <= float(self.fault.get(key,1.0 if key=="release_duty" else 0.0)) <= 1:
                raise ValueError(f"{key}必须在[0,1]")
        for key in ("sensor_tau_factor","sensor_noise_factor","servo_time_factor"):
            if float(self.fault.get(key,1.0)) <= 0:
                raise ValueError(f"{key}必须为正")
        if not math.isfinite(self.success_hold_s) or not math.isfinite(self.dr_spread) or self.success_hold_s < 0 or self.dr_spread < 0 or float(self.fault.get("release_period_s",8))<=0:
            raise ValueError("持续到源/随机化幅度非负，释放周期为正")
        delay = float(self.fault.get("action_delay_steps",0))
        if delay < 0 or not delay.is_integer():
            raise ValueError("action_delay_steps必须为非负整数")
        cfg["include_blank_age_observation"] = True
        cfg["servo_60deg_time_s"] = float(cfg.get("servo_60deg_time_s",.12))*float(self.fault.get("servo_time_factor",1))
        cfg["sensor_response_tau"] = float(cfg.get("sensor_response_tau",.6))*float(self.fault.get("sensor_tau_factor",1))
        cfg["sensor_recovery_tau"] = float(cfg.get("sensor_recovery_tau",3))*float(self.fault.get("sensor_tau_factor",1))
        cfg["sensor_noise_std"] = float(cfg.get("sensor_noise_std",.006))*float(self.fault.get("sensor_noise_factor",1))
        self._goal_dwell = 0.0
        self._previous_reached = False
        super().__init__(cfg)
        old = self.plume
        self.plume = ExperimentPuff(seed=int(cfg.get("seed",0)),source_position=cfg.get("source_position"),
            dt=self.dt,gas_field_mode=str(cfg.get("gas_field_mode","puff")),
            wind_speed_range=tuple(cfg.get("wind_speed_range",(.12,.20))),
            wind_sampling_mode=old.wind_sampling_mode,plume_overrides=cfg.get("plume_overrides"),
            world_bounds=(self.world_min,self.world_max))
        for key in ("wind_direction_range","source_distance_range","far_source_crosswind_range","require_source_in_world"):
            setattr(self.plume,key,getattr(old,key))
        self.plume.domain_randomization = self.domain_randomization
        self.plume.spread = self.dr_spread
        self.plume.fault = self.fault
        self.field = self.plume
        self._experiment_rng = np.random.default_rng(0)
        self._delayed = deque()
        self._last_command = np.array([self.stop_action,0,0])
        self._wind_changed = False

    def _randomize_sensors(self):
        super()._randomize_sensors()
        if self.dr_spread == 1.0:
            return
        for key,attr in [("response_tau","sensor_response_tau"),("recovery_tau","sensor_recovery_tau"),("noise_std","sensor_noise_std")]:
            nominal = self._sensor_nominal[key]
            setattr(self,attr,max(1e-8,nominal+self.dr_spread*(getattr(self,attr)-nominal)))
        self.sensor_baseline_drift_std *= self.dr_spread

    def reset(self, *, seed=None, options=None):
        self._goal_dwell = 0.0
        self._previous_reached = False
        self._wind_changed = False
        self._delayed.clear()
        self._last_command = np.array([self.stop_action,0,0])
        self._experiment_rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0),0x455850]))
        self.plume.episode_time_s = 0.0
        opts = dict(options or {})
        scene = dict(opts.get("scenario") or {})
        if "wind_speed" in self.fault:
            speed = float(self.fault["wind_speed"])
            if speed <= 0:
                raise ValueError("测试风速必须为正")
            scene["wind_speed"] = speed
            self.plume.wind_speed_clip = (max(.001,speed-.03),speed+.04)
        scene["source_strength"] = float(scene.get("source_strength",1))*float(self.fault.get("source_factor",1))
        opts["scenario"] = scene
        obs,info = super().reset(seed=seed,options=opts)
        self._previous_reached = float(info["distance_to_source"]) <= self.goal_radius
        if self.fault:
            fault_seed = int(self._experiment_rng.integers(0,2**31))
            self.left_sensor = FaultSensor(self.left_sensor,self.fault,"left",self.dt,fault_seed)
            self.right_sensor = FaultSensor(self.right_sensor,self.fault,"right",self.dt,fault_seed+1)
        return obs,info

    def _build_observation(self):
        if self.fixed_initial_sector is not None and self.step_count == 0:
            sector = int(self.fixed_initial_sector)
            self.whiskers.left_angle = self.whiskers._sector_angle(sector, side="left")
            self.whiskers.right_angle = self.whiskers._sector_angle(sector, side="right")
            self.last_left_sector = self.last_right_sector = sector
        obs = super()._build_observation().copy()
        if self.feature_mask == "raw_only":
            scale = self.observation_builder.config.scale
            clip = self.observation_builder.config.clip
            obs[0:2] = np.clip([self.left_sensor.value/scale,self.right_sensor.value/scale],-clip,clip)
        obs[FEATURE_MASKS[self.feature_mask]] = 0
        noise = float(self.fault.get("heading_noise_deg",0.0))
        if noise and hasattr(self,"_experiment_rng"):
            heading = self.robot_state.heading + math.radians(float(self._experiment_rng.normal(0,noise)))
            obs[12:14] = [math.cos(heading),math.sin(heading)]
        self._last_obs = obs
        return obs

    def _source_search_reward_components(self,distance,reached,oob,left,right,position_displacement,**kwargs):
        self._goal_dwell = self._goal_dwell+self.dt if reached and self._previous_reached else 0.0
        self._previous_reached = bool(reached)
        qualified = reached and self._goal_dwell+1e-12>=self.success_hold_s
        return super()._source_search_reward_components(distance,qualified,oob,left,right,position_displacement,**kwargs)

    def step(self, action):
        self.plume.episode_time_s = self.step_count*self.dt
        active = self.plume.episode_time_s >= float(self.fault.get("start_s",0))
        command = np.asarray(action,dtype=np.int64)
        if active:
            if not self._wind_changed and "wind_turn_deg" in self.fault:
                self.plume.wind_direction_base += math.radians(float(self.fault["wind_turn_deg"]))
                self._wind_changed = True
            delay = int(self.fault.get("action_delay_steps",0))
            if delay:
                if not self._delayed:
                    self._delayed.extend(self._last_command.copy() for _ in range(delay))
                self._delayed.append(command.copy())
                command = self._delayed.popleft()
            if self._experiment_rng.random() < float(self.fault.get("command_drop_probability",0)):
                command = self._last_command.copy()
        self._last_command = command.copy()
        obs,reward,terminated,truncated,info = super().step(command)
        success = bool(info["is_success"] and self._goal_dwell+1e-12>=self.success_hold_s)
        info["is_success"] = success
        info["success_hold_s"] = self._goal_dwell
        if self.trajectory:
            self.trajectory[-1]["is_success"] = success
        return obs,reward,bool(success or info["out_of_bounds"]),truncated,info

class ExperimentActionAdapter(MethodActionAdapter):
    """固定触须在环境首帧设置，避免reset重复推进有状态预处理。"""
    def reset(self, *, seed=None, options=None):
        self._scan_sector, self._scan_direction, self._scan_count = 0, 1, 0
        return self.env.reset(seed=seed, options=options)

def make_experiment_env(config, variant, *, seed=0, evaluating=False, fault=None):
    cfg = {**config,**variant.get("config",{}),"seed":int(seed)}
    cfg["plume_overrides"] = {**config.get("plume_overrides",{}),**variant.get("config",{}).get("plume_overrides",{})}
    cfg["feature_mask"] = variant.get("feature_mask","none")
    cfg["dr_spread"] = variant.get("dr_spread",1.0)
    cfg["fault"] = fault or {}
    if evaluating:
        # 正式场景的物理参数固定；不再次施加训练域随机化。
        cfg["domain_randomization"] = False
    method = variant.get("method","main")
    if method in ("b6_left","b6_right"):
        cfg["active_whisker_side"] = "left" if method=="b6_left" else "right"
    if method == "b2_fixed":
        cfg["fixed_initial_sector"] = int(variant.get("fixed_sector", 5))
    env = ExperimentMobileEnv(cfg)
    if method in ("b2_fixed","b3_scan","b6_left","b6_right"):
        if env.whiskers.sector_count != 10:
            raise ValueError("E4动作适配器仅支持10扇区；几何消融使用main独立训练")
        env = ExperimentActionAdapter(env,method,fixed_sector=int(variant.get("fixed_sector",5)),
                                  scan_hold=int(variant.get("scan_hold",2)))
    return ObservationHistory(env,int(variant.get("history_length",20)))
