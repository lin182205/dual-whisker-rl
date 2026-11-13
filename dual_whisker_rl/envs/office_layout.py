"""100平方米小型办公：前厅分流、公共走廊和独立功能房间。"""

SMALL_OFFICE = "small_office_v1"


def rectangle(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def small_office_regions():
    """区域按墙体中心线铺满10×10米，左右分隔错位且公共区为L形。"""
    return {
        "R1": rectangle(-5., -.8, -5., -1.8),
        "R2": rectangle(-5., -.8, -1.8, 1.4),
        "R3": rectangle(-5., -.8, 1.4, 5.),
        "R4": rectangle(.8, 5., -2., 1.8),
        "R5": rectangle(.8, 5., 1.8, 3.4),
        "R6": rectangle(.8, 5., 3.4, 5.),
        "reception": rectangle(-.8, 5., -5., -2.),
        "corridor": rectangle(-.8, .8, -2., 5.),
    }


def small_office_doors():
    """七个常开房门加前厅与走廊的开放口，不添加门扇碰撞物。"""
    specs = [
        ("D1", "vertical", (-.8, -3.5), ("R1", "reception")),
        ("D2", "vertical", (-.8, -.2), ("R2", "corridor")),
        ("D3", "vertical", (-.8, 3.7), ("R3", "corridor")),
        ("D4", "horizontal", (3.4, -2.), ("R4", "reception")),
        ("D5", "vertical", (.8, .5), ("R4", "corridor")),
        ("D6", "vertical", (.8, 2.6), ("R5", "corridor")),
        ("D7", "vertical", (.8, 4.2), ("R6", "corridor")),
    ]
    doors = [dict(id=name, axis=axis, center=list(c), regions=list(regions), width_m=1., kind="doorway")
             for name, axis, c, regions in specs]
    doors.append(dict(id="P1", axis="horizontal", center=[0., -2.],
                      regions=["reception", "corridor"], width_m=1.6, kind="open_passage"))
    return doors


def small_office_semantics():
    return {
        "R1": dict(name_zh="行政办公室", function="office", label_position=[-2.9, -3.4]),
        "R2": dict(name_zh="团队办公室", function="office", label_position=[-2.9, -.2]),
        "R3": dict(name_zh="安静办公室", function="office", label_position=[-2.9, 3.2]),
        "R4": dict(name_zh="会议室", function="meeting_room", label_position=[2.9, -.1]),
        "R5": dict(name_zh="茶水间", function="pantry", label_position=[2.9, 2.6]),
        "R6": dict(name_zh="卫生间", function="restroom", label_position=[2.9, 4.2]),
        "reception": dict(name_zh="接待前厅", function="reception", label_position=[2.1, -3.5]),
        "corridor": dict(name_zh="主走廊", function="circulation", label_position=[0., 1.4]),
    }
