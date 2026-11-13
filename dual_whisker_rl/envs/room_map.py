"""房间共用几何：碰撞、可见性、雷达与带安全间距的最短路径。"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from dual_whisker_rl.envs.room_layouts import (
    IRREGULAR_SIX_ROOMS, irregular_six_room_regions, irregular_six_room_doors,
    walls_from_regions, point_in_polygon,
)
from dual_whisker_rl.envs.office_layout import (
    SMALL_OFFICE, small_office_regions, small_office_doors, small_office_semantics,
)


class RoomMap:
    """矩形墙体地图；所有几何查询使用同一份米制坐标。"""

    VERSION = "four_rooms_v1"
    LAYOUTS = (VERSION, IRREGULAR_SIX_ROOMS, SMALL_OFFICE)

    def __init__(self, walls=None, bounds=(-5.0, 5.0), *, layout=VERSION):
        if layout not in self.LAYOUTS:
            raise ValueError(f"未知房间地图版本: {layout}")
        self.layout = layout
        self.bounds = tuple(float(v) for v in bounds)
        self.regions, self.doors = {}, []
        if layout in (IRREGULAR_SIX_ROOMS, SMALL_OFFICE):
            if walls is not None or self.bounds != (-5., 5.):
                raise ValueError("固定房间布局使用自身墙体和[-5,5]场地")
            self.regions = small_office_regions() if layout == SMALL_OFFICE else irregular_six_room_regions()
            self.doors = small_office_doors() if layout == SMALL_OFFICE else irregular_six_room_doors()
            walls = walls_from_regions(self.regions, self.doors, self.bounds)
        elif walls is None:
            spans = [(-5.0, -3.0), (-2.0, 2.0), (3.0, 5.0)]
            walls = [(-0.05, 0.05, lo, hi) for lo, hi in spans]
            walls += [(lo, hi, -0.05, 0.05) for lo, hi in spans]
        self.walls = np.asarray(walls, dtype=float).reshape(-1, 4)
        if not np.isfinite(self.walls).all() or any(
            r[0] >= r[1] or r[2] >= r[3] for r in self.walls
        ):
            raise ValueError("墙体必须是有效的有限矩形")
        self._graphs = {}

    def metadata(self):
        if self.layout == SMALL_OFFICE:
            semantics = small_office_semantics()
            return {"layout": self.layout, "world_bounds": list(self.bounds),
                    "walls": self.walls.tolist(), "wall_thickness_m": .1,
                    "door_width_m": 1., "room_count": 6, "region_count": 8,
                    "corridor_clear_width_m": 1.5,
                    "exterior_entrance": {"center": [2.6, -5.], "width_m": 1.2,
                                          "state": "closed", "role": "boundary_annotation"},
                    "regions": [{"id": name, "kind": "room" if name.startswith("R") else "common_area",
                                 "polygon": [list(p) for p in polygon], **semantics[name]}
                                for name, polygon in self.regions.items()],
                    "doors": self.doors,
                    "description": "small office with reception, circulation, three offices, meeting, pantry and restroom"}
        if self.layout == IRREGULAR_SIX_ROOMS:
            return {"layout": self.layout, "world_bounds": list(self.bounds),
                    "walls": self.walls.tolist(), "wall_thickness_m": .1,
                    "door_width_m": 1., "room_count": 6,
                    "regions": [{"id": name, "kind": "corridor" if name == "corridor" else "room",
                                 "polygon": [list(p) for p in polygon]}
                                for name, polygon in self.regions.items()],
                    "doors": [{**d, "center": list(d["center"]), "regions": list(d["regions"])}
                              for d in self.doors]}
        # 四房间元数据逐字段保留，已保存模型和场景库的地图摘要不变。
        return {"layout": self.VERSION, "world_bounds": list(self.bounds),
                "walls": self.walls.tolist(), "wall_thickness_m": 0.1,
                "door_width_m": 1.0, "door_centers_m": [-2.5, 2.5]}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()

    def valid(self, points, margin=0.0):
        """半径膨胀采用保守矩形，和导航图采用相同安全包络。"""
        p = np.asarray(points, dtype=float)
        lo, hi = self.bounds
        ok = np.isfinite(p).all(axis=-1) & (p > lo + margin).all(axis=-1) & (p < hi - margin).all(axis=-1)
        for x0, x1, y0, y1 in self.walls:
            ok &= ~((p[..., 0] >= x0-margin) & (p[..., 0] <= x1+margin)
                    & (p[..., 1] >= y0-margin) & (p[..., 1] <= y1+margin))
        return ok

    def first_hit(self, start, end, margin=0.0):
        """批量线段求交，返回首次命中比例和朝自由区域的法向量。"""
        a, b = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
        d = b-a
        shape = a.shape[:-1]
        best = np.full(shape, np.inf)
        normal = np.zeros_like(a)
        lo, hi = self.bounds
        for axis in (0, 1):
            for bound, sign in ((lo+margin, 1.0), (hi-margin, -1.0)):
                moving_out = d[..., axis] * sign < 0
                t = np.divide(bound-a[..., axis], d[..., axis],
                              out=np.full(shape, np.inf), where=moving_out)
                hit = moving_out & (t >= 0) & (t <= 1) & (t < best)
                best = np.where(hit, t, best)
                n = np.zeros_like(a); n[..., axis] = sign
                normal = np.where(hit[..., None], n, normal)
        for x0, x1, y0, y1 in self.walls:
            lower = np.array([x0-margin, y0-margin])
            upper = np.array([x1+margin, y1+margin])
            stationary = np.abs(d) < 1e-15
            safe_d = np.where(stationary, 1.0, d)
            t0, t1 = (lower-a)/safe_d, (upper-a)/safe_d
            near = np.where(stationary, -np.inf, np.minimum(t0, t1))
            far = np.where(stationary, np.inf, np.maximum(t0, t1))
            enter, leave = near.max(axis=-1), far.min(axis=-1)
            parallel_outside = (stationary & ((a < lower) | (a > upper))).any(axis=-1)
            t = np.maximum(enter, 0.0)
            hit = ~parallel_outside & (leave >= t) & (t <= 1) & (t < best)
            axis = np.argmax(near, axis=-1)
            n = np.zeros_like(a)
            for k in (0, 1):
                n[..., k] = np.where(axis == k, np.where(d[..., k] > 0, -1., 1.), 0.)
            best = np.where(hit, t, best)
            normal = np.where(hit[..., None], n, normal)
        return best, normal

    def visible(self, start, end, margin=0.0):
        t, _ = self.first_hit(start, end, margin)
        return ~np.isfinite(t)

    def clip_sample(self, start, end):
        a, b = np.asarray(start, float), np.asarray(end, float)
        t, _ = self.first_hit(a, b)
        if np.isfinite(t):
            b = a + max(0., float(t)-1e-6/max(np.linalg.norm(b-a), 1e-12))*(b-a)
        return tuple(float(v) for v in b)

    def reflect(self, start, end):
        """对剩余位移连续反射；多次碰撞也不会跳过薄墙。"""
        a, b = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
        a, b = a.copy(), b.copy()
        for _ in range(12):
            t, n = self.first_hit(a, b)
            hit = np.isfinite(t)
            if not np.any(hit):
                return b
            f = np.where(hit, t, 1.)[..., None]
            contact = a + f*(b-a)
            remaining = (1-f)*(b-a)
            reflected = remaining-2*(remaining*n).sum(axis=-1)[..., None]*n
            a = np.where(hit[..., None], contact+1e-8*n, b)
            b = np.where(hit[..., None], a+reflected, b)
        # 极端大位移达到反射上限时停在最后一个安全点。
        return a

    def raycast(self, position, heading, beams=36, max_range=3.0):
        angles = heading + np.arange(beams)*2*np.pi/beams
        end = np.asarray(position) + max_range*np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        t, _ = self.first_hit(position, end)
        return (np.minimum(t, 1.)*max_range).astype(np.float32)

    def room_id(self, point):
        if self.regions:
            if not self.valid(point):
                return None
            # 门槛优先归走廊，房间之间共边按稳定编号顺序归属。
            priority = ["corridor"] + (["reception"] if self.layout == SMALL_OFFICE else [])
            for name in priority + [f"R{i}" for i in range(1, 7)]:
                if point_in_polygon(point, self.regions[name]):
                    return name
            raise RuntimeError("有效自由位置未被房间或走廊覆盖")
        x, y = point
        return ("north" if y >= 0 else "south") + ("_east" if x >= 0 else "_west")

    def sample_free(self, rng, margin):
        lo, hi = self.bounds
        for _ in range(10000):
            p = rng.uniform(lo+margin, hi-margin, size=2)
            if self.valid(p, margin):
                return tuple(float(v) for v in p)
        raise RuntimeError("无法在房间地图中采样有效位置")

    def navigation_graph(self, radius):
        """缓存膨胀墙体的可视图；角点略移出墙面以消除擦边歧义。"""
        if radius not in self._graphs:
            vertices = []
            eps = 1e-6
            for x0, x1, y0, y1 in self.walls:
                for x in (x0-radius-eps, x1+radius+eps):
                    for y in (y0-radius-eps, y1+radius+eps):
                        if self.valid((x, y), radius):
                            vertices.append((x, y))
            v = (self._union_navigation_vertices(radius) if self.regions
                 else np.unique(np.asarray(vertices).reshape(-1, 2), axis=0))
            d = np.linalg.norm(v[:, None]-v[None, :], axis=-1)
            d = np.where(self.visible(v[:, None], v[None, :], radius), d, np.inf)
            np.fill_diagonal(d, 0.)
            for k in range(len(v)):
                d = np.minimum(d, d[:, k, None]+d[None, k, :])
            self._graphs[radius] = (v, d)
        return self._graphs[radius]

    def _union_navigation_vertices(self, radius):
        """矩形并集边界角点包含边相交形成的角，不能只筛选原矩形的四角。"""
        expanded = self.walls + np.array([-radius, radius, -radius, radius])
        candidates = set()
        for x0, x1, y0, y1 in expanded:
            for x in (x0, x1):
                for y in (y0, y1):
                    candidates.add((round(float(x), 12), round(float(y), 12)))
                for a0, a1, b0, b1 in expanded:
                    if a0-1e-10 <= x <= a1+1e-10:
                        for y in (b0, b1):
                            if y0-1e-10 <= y <= y1+1e-10:
                                candidates.add((round(float(x), 12), round(float(y), 12)))
        if not candidates:
            return np.empty((0, 2))
        points = np.asarray(sorted(candidates))
        offsets = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        eps = 1e-6
        free = self.valid(points[:, None, :]+eps*offsets, radius)
        count = free.sum(axis=1)
        corners = (count == 1) | (count == 3)
        directions = free.astype(float) @ offsets
        vertices = points[corners] + eps*directions[corners]
        return np.unique(vertices[self.valid(vertices, radius)], axis=0)

    def source_distances(self, source, radius):
        v, graph = self.navigation_graph(radius)
        direct = np.where(self.visible(v, source, radius), np.linalg.norm(v-source, axis=-1), np.inf)
        return np.min(graph+direct[None, :], axis=1) if len(v) else direct

    def path_distance(self, point, source, radius, source_distances=None):
        if not self.valid(point, radius) or not self.valid(source, radius):
            return math.inf
        if self.visible(point, source, radius):
            return float(np.linalg.norm(np.asarray(point)-source))
        v, _ = self.navigation_graph(radius)
        if not len(v):
            return math.inf
        sd = self.source_distances(source, radius) if source_distances is None else source_distances
        distances = np.where(self.visible(point, v, radius), np.linalg.norm(v-point, axis=-1), np.inf)
        return float(np.min(distances+sd))
