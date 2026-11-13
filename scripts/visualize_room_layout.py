"""从训练地图几何生成带功能、面积和尺寸的室内平面图。"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.font_utils import resolve_chinese_font
from dual_whisker_rl.paths import project_path


def render_layout(room_map, path):
    """图中墙与门洞直接使用环境矩形；不额外画装饰性门扇障碍。"""
    meta = room_map.metadata()
    regular = FontProperties(fname=str(resolve_chinese_font()))
    bold = FontProperties(fname=str(resolve_chinese_font(bold=True)))
    fig, ax = plt.subplots(figsize=(10, 11), facecolor="#fafbfd")
    ax.set_facecolor("white")
    colors = {"office": "#eaf1fa", "meeting_room": "#e0eafa", "pantry": "#e6f2ef",
              "restroom": "#eef0f5", "reception": "#e4f0f4", "circulation": "#f6f8fc"}
    for region in meta["regions"]:
        poly = np.asarray(region["polygon"])
        ax.add_patch(Polygon(poly, facecolor=colors.get(region.get("function"), "#eef3fa"),
                             edgecolor="none", zorder=0))
        area = abs(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1)))/2.
        p = region.get("label_position", poly.mean(axis=0))
        if region["id"] == "corridor":
            label = "主\n走\n廊\n\n净宽\n1.50m"
        else:
            label = f"{region['name_zh']}\n{region['id']}   {area:.2f} m²"
        ax.text(*p, label, ha="center", va="center", fontproperties=regular, fontsize=12,
                color="#263e59", linespacing=1.7)
    for x0, x1, y0, y1 in room_map.walls:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor="#25384b", zorder=3))
    ax.add_patch(Rectangle((-5, -5), 10, 10, fill=False, linewidth=4, edgecolor="#25384b", zorder=3))
    if 'exterior_entrance' in meta:
        entrance = meta['exterior_entrance']
        x,y = entrance['center']
        ax.plot([x-entrance['width_m']/2,x+entrance['width_m']/2],[y,y],lw=4,color='#5995a2',zorder=4)
        ax.text(x,y-.16,'主入口（关闭）',ha='center',va='top',fontproperties=regular,fontsize=9,color='#397482')
    for door in room_map.doors:
        x, y = door['center']
        vertical = door['axis'] == 'vertical'
        if vertical:
            ax.plot([x, x], [y-door['width_m']/2, y+door['width_m']/2], color="#3e8390", lw=2, alpha=.6)
        else:
            ax.plot([x-door['width_m']/2, x+door['width_m']/2], [y, y], color="#3e8390", lw=2, alpha=.6)
        if door.get('kind') == 'open_passage':
            ax.text(x, y-.2, "开放口 1.6m", fontproperties=regular, ha='center', va='top', fontsize=9, color="#3c7883")
        else:
            ax.annotate(door['id'], xy=(x, y), xytext=(9, 0) if vertical else (0, 8),
                        textcoords='offset points', ha='left' if vertical else 'center', va='center',
                        fontsize=8, color="#347a85", zorder=5)
    for a, b in [((-5, 5.45), (5, 5.45)), ((-5.45, -5), (-5.45, 5))]:
        ax.annotate('', xy=a, xytext=b, arrowprops=dict(arrowstyle='<->', color='#72869c', lw=1))
    ax.text(0, 5.55, '10.00 m', ha='center', va='bottom', fontsize=10, color='#536a83')
    ax.text(-5.6, 0, '10.00 m', ha='center', va='center', rotation=90, fontsize=10, color='#536a83')
    ax.set(xlim=(-6, 5.6), ylim=(-5.6, 6), aspect='equal')
    ax.axis('off')
    fig.suptitle("100 m² 小型办公平面布局", fontproperties=bold, fontsize=22, color="#20384f", y=.96)
    fig.text(.5, .915, "前厅分流 · 公共走廊 · 六个功能房间", ha='center', fontproperties=regular,
             fontsize=12, color='#63788d')
    fig.text(.5, .065, "房门宽 1.00m  |  墙厚 0.10m  |  房门保持开启，仅墙体参与碰撞", ha='center',
             fontproperties=regular, fontsize=10, color='#536a83')
    fig.text(.5, .04, "面积按墙体中心线划分；气源和机器人可在全部有效区域随机初始化", ha='center',
             fontproperties=regular, fontsize=9, color='#78899b')
    fig.subplots_adjust(left=.03, right=.98, bottom=.09, top=.88)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/mobile_rooms_hybrid.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/figures/office_floorplan.png'))
    args = parser.parse_args()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8'))
    room_map = RoomMap(layout=cfg['map_layout'])
    render_layout(room_map, project_path(args.output))
    print(f"layout={room_map.layout} map_sha256={room_map.digest} output={project_path(args.output)}")


if __name__ == '__main__':
    main()
