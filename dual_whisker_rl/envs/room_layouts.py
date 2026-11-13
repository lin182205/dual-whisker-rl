"""固定房间区域与门洞；区域边界采用墙体中心线坐标。"""

from __future__ import annotations

from collections import defaultdict


IRREGULAR_SIX_ROOMS = "irregular_six_rooms_v1"


def irregular_six_room_regions():
    """六个大小不一的房间及独立 L 形短走廊，共同铺满10×10场地。"""
    return {
        "R1": [(-5., -5.), (-1.4, -5.), (-1.4, -.8), (-5., -.8)],
        "R2": [(-5., -.8), (-.6, -.8), (-.6, 1.5), (-5., 1.5)],
        "R3": [(-5., 1.5), (1., 1.5), (1., 5.), (-5., 5.)],
        "R4": [(-1.4, -5.), (5., -5.), (5., -2.), (-1.4, -2.)],
        "R5": [(.6, -2.), (5., -2.), (5., .3), (.6, .3)],
        "R6": [(.6, .3), (5., .3), (5., 5.), (1., 5.), (1., 1.5), (.6, 1.5)],
        "corridor": [(-1.4, -2.), (.6, -2.), (.6, 1.5), (-.6, 1.5),
                     (-.6, -.8), (-1.4, -.8)],
    }


def irregular_six_room_doors():
    """门洞的轴向表示所在墙体方向；八个门均宽1米。"""
    definitions = [
        ("D1", "vertical", (-1.4, -1.4), ("R1", "corridor")),
        ("D2", "vertical", (-.6, .2), ("R2", "corridor")),
        ("D3", "horizontal", (0., 1.5), ("R3", "corridor")),
        ("D4", "horizontal", (-.2, -2.), ("R4", "corridor")),
        ("D5", "vertical", (.6, -1.), ("R5", "corridor")),
        ("D6", "vertical", (.6, .9), ("R6", "corridor")),
        ("D7", "vertical", (-1.4, -3.5), ("R1", "R4")),
        ("D8", "vertical", (1., 3.2), ("R3", "R6")),
    ]
    return [dict(id=name, axis=axis, center=list(center), regions=list(regions), width_m=1.)
            for name, axis, center, regions in definitions]


def walls_from_regions(regions, doors, bounds=(-5., 5.), thickness=.1):
    """合并共线区域边界，扣除门洞后生成矩形墙；外墙由场地边界负责。"""
    lines = defaultdict(list)
    for polygon in regions.values():
        for p, q in zip(polygon, polygon[1:]+polygon[:1]):
            vertical = p[0] == q[0]
            axis = "vertical" if vertical else "horizontal"
            constant = p[0] if vertical else p[1]
            if constant in bounds:
                continue
            k = 1 if vertical else 0
            lines[(axis, constant)].append(tuple(sorted((p[k], q[k]))))
    walls = []
    for (axis, constant), intervals in sorted(lines.items()):
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        for door in doors:
            x, y = door["center"]
            dc, middle = (x, y) if axis == "vertical" else (y, x)
            if door["axis"] != axis or dc != constant:
                continue
            a, b = middle-door["width_m"]/2., middle+door["width_m"]/2.
            pieces = []
            for lo, hi in merged:
                if hi <= a or lo >= b:
                    pieces.append([lo, hi])
                else:
                    if lo < a:
                        pieces.append([lo, a])
                    if hi > b:
                        pieces.append([b, hi])
            merged = pieces
        for lo, hi in merged:
            half = thickness/2.
            walls.append((constant-half, constant+half, lo, hi) if axis == "vertical"
                         else (lo, hi, constant-half, constant+half))
    return walls


def point_in_polygon(point, polygon):
    """含边界的射线法；共享边界由调用者的固定区域顺序消除歧义。"""
    x, y = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross) <= 1e-10 and min(ax, bx)-1e-10 <= x <= max(ax, bx)+1e-10 \
                and min(ay, by)-1e-10 <= y <= max(ay, by)+1e-10:
            return True
        if (ay > y) != (by > y) and x < ax+(y-ay)*(bx-ax)/(by-ay):
            inside = not inside
    return inside
