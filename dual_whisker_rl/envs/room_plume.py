"""带墙体边界和浓度遮挡的轻量动态 puff；不求解室内 CFD 风场。"""

from __future__ import annotations

import math

import numpy as np

from dual_whisker_rl.envs.whisker_only_env import DynamicPuffPlume


class RoomPuffPlume(DynamicPuffPlume):
    """复用动态羽流参数与随机过程，只替换几何相关步骤。"""

    def __init__(self, room_map, source_clearance=0.25, **kwargs):
        self.room_map = room_map
        super().__init__(obstacles=room_map.walls, **kwargs)
        self.source_clearance = float(source_clearance)

    def is_position_valid(self, x, y, margin=0.):
        return bool(self.room_map.valid((x, y), margin))

    def is_occupied(self, x, y):
        return not self.is_position_valid(x, y)

    def _place_source_upwind(self):
        # 房间场景的源位置不再由风向决定。
        # 固定场景仍消费同一份初始化随机数，封存后可逐步复现原采样局。
        sampled = self.room_map.sample_free(self.rng, self.source_clearance)
        source = sampled if self.source_position_override is None else self.source_position_override
        if not self.room_map.valid(source, self.source_clearance):
            raise ValueError("房间气源位置无效或距墙过近")
        self.source_x, self.source_y = (float(v) for v in source)

    def _emit_puffs(self):
        super()._emit_puffs()
        # 释放抖动也不能把气团直接生在墙内或墙的另一侧。
        born = [p for p in self.puffs if p["age"] == 0.0]
        if born:
            positions = self.room_map.reflect(
                (self.source_x, self.source_y), [(p["x"], p["y"]) for p in born]
            )
            for p, xy in zip(born, positions):
                p["x"], p["y"] = (float(v) for v in xy)

    def _advance_puffs(self):
        if not self.puffs:
            return
        keys = ("x", "y", "vd", "vc", "cos_direction", "sin_direction")
        values = np.asarray([[p[k] for k in keys] for p in self.puffs])
        positions = values[:, :2]
        # 按气团交替抽取 vd/vc，与原 per-puff OU 更新顺序一致。
        noise = self.rng.normal(size=(len(values), 2))
        vd = self.turbulence_vel_relax*values[:, 2] + self.turbulence_vel_std_downwind*noise[:, 0]
        vc = self.turbulence_vel_relax*values[:, 3] + self.turbulence_vel_std_crosswind*noise[:, 1]
        c, s = values[:, 4], values[:, 5]
        velocity = np.column_stack((
            self.wind_speed*math.cos(self.wind_direction)+vd*c-vc*s,
            self.wind_speed*math.sin(self.wind_direction)+vd*s+vc*c,
        ))
        positions = self.room_map.reflect(positions, positions+velocity*self.dt)
        active = []
        for i, p in enumerate(self.puffs):
            p["x"], p["y"] = (float(v) for v in positions[i])
            p["vd"], p["vc"] = float(vd[i]), float(vc[i])
            p["age"] += self.dt
            p["sigma_downwind"] += self.diffusion_downwind_rate*self.dt
            p["sigma_crosswind"] += self.diffusion_crosswind_rate*self.dt
            p["mass"] *= math.exp(-self.decay_rate*self.dt)
            if p["age"] <= self.max_puff_age and p["mass"] >= self.min_puff_mass:
                active.append(p)
        self.puffs = active[-self.max_puffs:]

    def _concentrations(self, points, add_noise=False):
        """全部采样入口共享同一公式；无噪声绘图不消耗任何随机数。"""
        points = np.asarray(points, float).reshape(-1, 2)
        result = np.zeros(len(points), dtype=float)
        valid = self.room_map.valid(points)
        core = np.asarray(self._source_core_concentration(points[:, 0], points[:, 1]), dtype=float)
        result[valid] = self.gas_background + core[valid]*self.room_map.visible(
            (self.source_x, self.source_y), points[valid]
        )
        if self.puffs and np.any(valid):
            data = np.asarray([[p[k] for k in ("x", "y", "cos_direction", "sin_direction",
                                              "sigma_downwind", "sigma_crosswind", "mass")]
                               for p in self.puffs])
            ids = np.flatnonzero(valid)
            for start in range(0, len(ids), 128):
                idx = ids[start:start+128]
                q = points[idx]
                dx, dy = q[None, :, 0]-data[:, None, 0], q[None, :, 1]-data[:, None, 1]
                down = dx*data[:, None, 2]+dy*data[:, None, 3]
                cross = -dx*data[:, None, 3]+dy*data[:, None, 2]
                exponent = -.5*((down/data[:, None, 4])**2+(cross/data[:, None, 5])**2)
                # 同一点/双点/网格使用相同数值截断，忽略远小于噪声的尾部。
                pi, qi = np.nonzero(exponent > -32.)
                if len(pi):
                    visible = self.room_map.visible(data[pi, :2], q[qi])
                    contribution = data[pi, 6]*np.exp(exponent[pi, qi])/(2*np.pi*data[pi, 4]*data[pi, 5])
                    result[idx] += np.bincount(qi, weights=contribution*visible, minlength=len(q))
        if add_noise:
            result[valid] += self.noise_rng.normal(0., self.gas_noise_std, int(valid.sum()))
        return np.maximum(result, 0.)

    def concentration(self, x, y, add_noise=True):
        return float(self._concentrations([(x, y)], add_noise)[0])

    def concentration_pair(self, left_point, right_point, add_noise=True):
        return tuple(float(v) for v in self._concentrations([left_point, right_point], add_noise))

    def grid(self, resolution=80, add_noise=False):
        xs = np.linspace(self.world_min, self.world_max, resolution, dtype=np.float32)
        x, y = np.meshgrid(xs, xs)
        field = self._concentrations(np.column_stack((x.ravel(), y.ravel())), add_noise)
        return xs, xs.copy(), field.reshape(x.shape).astype(np.float32)

    def metadata(self):
        return {**super().metadata(), "room_map": self.room_map.metadata(),
                "map_sha256": self.room_map.digest, "gas_field_mode": "room_puff",
                "boundary_model": "reflected_centers_visibility_mask_v1"}
