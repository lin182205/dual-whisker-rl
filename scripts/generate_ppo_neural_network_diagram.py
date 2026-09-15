"""生成默认 MobileWhisker GRU-PPO 的神经元连接结构图。"""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


WIDTH = 3400
HEIGHT = 2000
BACKGROUND = "#0b0f14"
TEXT = "#f1f5f9"
MUTED = "#aeb8c6"
STRUCTURE = "#334155"
LINE = "#aeb8c6"
INPUT_COLOR = "#58c7df"
GRU_COLOR = "#55d6be"
FEATURE_COLOR = "#70a7ff"
ACTOR_COLOR = "#68d6ff"
MOVE_COLOR = "#58c7df"
LEFT_COLOR = "#79d98c"
RIGHT_COLOR = "#f2aa63"
CRITIC_COLOR = "#bd91ff"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.font_utils import resolve_chinese_font

FIGURE_DIR = ROOT / "docs" / "figures"
PNG_PATH = FIGURE_DIR / "algorithm-ppo-neural-network.png"
SVG_PATH = FIGURE_DIR / "algorithm-ppo-neural-network.svg"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(resolve_chinese_font(bold=bold)), size=size)


def text_png(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    size: int,
    fill: str = TEXT,
    anchor: str = "mm",
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill, anchor=anchor)


def node_png(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: str,
    *,
    radius: int = 16,
) -> None:
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=BACKGROUND,
        outline=color,
        width=5,
    )


def bracket_png(
    draw: ImageDraw.ImageDraw,
    x: int,
    y1: int,
    y2: int,
    color: str,
    *,
    facing: str = "right",
) -> None:
    direction = 1 if facing == "right" else -1
    draw.line(
        (
            x + direction * 24,
            y1,
            x,
            y1 + 18,
            x,
            y2 - 18,
            x + direction * 24,
            y2,
        ),
        fill=color,
        width=5,
    )


def text_svg(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    fill: str = TEXT,
    anchor: str = "middle",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'dominant-baseline="middle" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def node_svg(x: int, y: int, color: str, *, radius: int = 16) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{BACKGROUND}" '
        f'stroke="{color}" stroke-width="5"/>'
    )


def compressed_nodes(top: int, bottom: int, count: int, step: int) -> list[int]:
    return [top + step * i for i in range(count)] + [
        bottom + step * i for i in range(count)
    ]


def build() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_input, x_gru, x_feature = 300, 710, 1120
    x_hidden_1, x_hidden_2, x_output = 1670, 2180, 2670

    shared_y = compressed_nodes(520, 1320, 6, 45)
    actor_y = compressed_nodes(300, 740, 7, 42)
    critic_y = compressed_nodes(1220, 1580, 6, 42)
    actor_output_y = [310 + 27 * i for i in range(26)]
    critic_output_y = [1455]

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    connections = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(connections)

    def connect(
        x1: int,
        ys1: list[int],
        x2: int,
        ys2: list[int],
        alpha: int,
    ) -> None:
        for y1 in ys1:
            for y2 in ys2:
                line_draw.line(
                    (x1 + 18, y1, x2 - 18, y2),
                    fill=(174, 184, 198, alpha),
                    width=1,
                )

    connect(x_input, shared_y, x_gru, shared_y, 55)
    connect(x_gru, shared_y, x_feature, shared_y, 48)
    connect(x_feature, shared_y, x_hidden_1, actor_y, 38)
    connect(x_hidden_1, actor_y, x_hidden_2, actor_y, 46)
    connect(x_hidden_2, actor_y, x_output, actor_output_y, 38)
    connect(x_feature, shared_y, x_hidden_1, critic_y, 38)
    connect(x_hidden_1, critic_y, x_hidden_2, critic_y, 46)
    connect(x_hidden_2, critic_y, x_output, critic_output_y, 55)

    image = Image.alpha_composite(image.convert("RGBA"), connections).convert("RGB")
    draw = ImageDraw.Draw(image)

    text_png(draw, (WIDTH / 2, 54), "PPO Actor–Critic 网络结构", size=48, bold=True)
    text_png(
        draw,
        (WIDTH / 2, 108),
        "默认障碍场 GRU 配置  ·  历史长度 20  ·  单帧观测 28 维",
        size=26,
        fill=MUTED,
    )

    draw.line((1420, 155, 2040, 155), fill=ACTOR_COLOR, width=3)
    draw.line((2390, 155, 3010, 155), fill=ACTOR_COLOR, width=3)
    text_png(draw, (2215, 155), "Actor 策略分支", size=30, fill=ACTOR_COLOR, bold=True)
    draw.line((1420, 1080, 2030, 1080), fill=CRITIC_COLOR, width=3)
    draw.line((2400, 1080, 3010, 1080), fill=CRITIC_COLOR, width=3)
    text_png(draw, (2215, 1080), "Critic 价值分支", size=30, fill=CRITIC_COLOR, bold=True)

    shared_headers = [
        (x_input, "历史观测序列", "B × 20 × 28", "由 B × 560 重塑"),
        (x_gru, "共享 GRU", "1 层，隐藏宽度 64", "最终隐藏状态：B × 64"),
        (x_feature, "共享特征投影", "Linear 64 → 64 + GELU", "输出特征：B × 64"),
    ]
    for x, title, shape, detail in shared_headers:
        text_png(draw, (x, 355), title, size=28, bold=True)
        text_png(draw, (x, 394), shape, size=23, fill=MUTED)
        text_png(draw, (x, 429), detail, size=20, fill=MUTED)

    actor_headers = [
        (x_hidden_1, "Actor 隐藏层 1", "Linear 64 → 160 + Tanh", "B × 160  ·  10,400 参数"),
        (x_hidden_2, "Actor 隐藏层 2", "Linear 160 → 160 + Tanh", "B × 160  ·  25,760 参数"),
        (x_output, "动作输出层", "Linear 160 → 26", "B × 26 logits  ·  4,186 参数"),
    ]
    for x, title, shape, detail in actor_headers:
        text_png(draw, (x, 202), title, size=27, fill=ACTOR_COLOR, bold=True)
        text_png(draw, (x, 238), shape, size=22, fill=MUTED)
        text_png(draw, (x, 268), detail, size=19, fill=MUTED)

    critic_headers = [
        (x_hidden_1, "Critic 隐藏层 1", "Linear 64 → 160 + Tanh", "B × 160  ·  10,400 参数"),
        (x_hidden_2, "Critic 隐藏层 2", "Linear 160 → 160 + Tanh", "B × 160  ·  25,760 参数"),
        (x_output, "价值输出层", "Linear 160 → 1", "状态价值 V(S)：B × 1  ·  161 参数"),
    ]
    for x, title, shape, detail in critic_headers:
        text_png(draw, (x, 1122), title, size=27, fill=CRITIC_COLOR, bold=True)
        text_png(draw, (x, 1158), shape, size=22, fill=MUTED)
        text_png(draw, (x, 1188), detail, size=19, fill=MUTED)

    for y in shared_y:
        node_png(draw, x_input, y, INPUT_COLOR)
        node_png(draw, x_gru, y, GRU_COLOR)
        node_png(draw, x_feature, y, FEATURE_COLOR)
    for y in actor_y:
        node_png(draw, x_hidden_1, y, ACTOR_COLOR)
        node_png(draw, x_hidden_2, y, ACTOR_COLOR)
    for y in critic_y:
        node_png(draw, x_hidden_1, y, CRITIC_COLOR)
        node_png(draw, x_hidden_2, y, CRITIC_COLOR)

    for x, ys in (
        (x_input, (910, 960, 1010)),
        (x_gru, (910, 960, 1010)),
        (x_feature, (910, 960, 1010)),
        (x_hidden_1, (626, 656, 686)),
        (x_hidden_2, (626, 656, 686)),
        (x_hidden_1, (1470, 1500, 1530)),
        (x_hidden_2, (1470, 1500, 1530)),
    ):
        for y in ys:
            text_png(draw, (x, y), "·", size=38, fill=MUTED)

    output_colors = [MOVE_COLOR] * 6 + [LEFT_COLOR] * 10 + [RIGHT_COLOR] * 10
    output_indices = list(range(6)) + list(range(10)) + list(range(10))
    for y, color, index in zip(actor_output_y, output_colors, output_indices):
        node_png(draw, x_output, y, color, radius=14)
        text_png(draw, (x_output + 34, y), str(index), size=17, fill=MUTED, anchor="lm")

    node_png(draw, x_output, critic_output_y[0], CRITIC_COLOR, radius=25)
    text_png(draw, (x_output, critic_output_y[0]), "V", size=22, fill=CRITIC_COLOR, bold=True)

    bracket_png(draw, 196, 500, 1590, TEXT)
    text_png(draw, (110, 995), "28", size=52, bold=True)
    text_png(draw, (110, 1048), "维／帧", size=23, fill=MUTED)
    text_png(draw, (110, 1091), "× 20 帧", size=23, fill=INPUT_COLOR, bold=True)

    action_groups = [
        (292, 459, MOVE_COLOR, "移动动作", "6 个 logits"),
        (459, 729, LEFT_COLOR, "左触须扇区", "10 个 logits"),
        (729, 999, RIGHT_COLOR, "右触须扇区", "10 个 logits"),
    ]
    for y1, y2, color, label, dim in action_groups:
        bracket_png(draw, 2795, y1, y2, color)
        center = (y1 + y2) / 2
        text_png(draw, (2842, center - 17), label, size=24, fill=color, anchor="lm", bold=True)
        text_png(draw, (2842, center + 20), dim, size=19, fill=MUTED, anchor="lm")
    text_png(draw, (3030, 637), "三组分别形成\nCategorical 分布", size=22, fill=MUTED, anchor="lm")

    text_png(draw, (x_gru, 1605), "沿 20 帧递推，取最后隐藏状态", size=21, fill=GRU_COLOR)
    text_png(draw, (x_gru, 1641), "GRU：18,048 参数", size=20, fill=MUTED)
    text_png(draw, (x_feature, 1605), "共享给 Actor 与 Critic", size=21, fill=FEATURE_COLOR)
    text_png(draw, (x_feature, 1641), "投影层：4,160 参数", size=20, fill=MUTED)
    text_png(draw, (x_output + 65, 1512), "V(S)", size=25, fill=CRITIC_COLOR, anchor="lm", bold=True)
    text_png(draw, (x_output + 65, 1548), "每个样本一个价值标量", size=20, fill=MUTED, anchor="lm")

    draw.line((200, 1810, 3200, 1810), fill=STRUCTURE, width=2)
    text_png(
        draw,
        (WIDTH / 2, 1855),
        "共享编码器 22,208 参数  ·  Actor 40,346 参数  ·  Critic 36,321 参数  ·  总计 98,875 参数",
        size=25,
        fill=MUTED,
    )
    text_png(
        draw,
        (WIDTH / 2, 1900),
        "B 为批量大小；无障碍配置仅把输入改为 B × 20 × 16（即 B × 320），其余网络层保持不变",
        size=22,
        fill=MUTED,
    )
    image.save(PNG_PATH, format="PNG", optimize=True)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<title>PPO Actor-Critic 网络结构</title>",
        "<desc>默认障碍场 GRU-PPO 网络。二十帧、每帧二十八维观测由共享 GRU 编码成六十四维特征，再进入两个独立的一百六十维双层网络，Actor 输出二十六个动作 logits，Critic 输出一个状态价值。</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        '<g fill="none" stroke-linecap="round">',
    ]

    def svg_connect(
        x1: int,
        ys1: list[int],
        x2: int,
        ys2: list[int],
        opacity: float,
    ) -> None:
        for y1 in ys1:
            for y2 in ys2:
                svg.append(
                    f'<line x1="{x1 + 18}" y1="{y1}" x2="{x2 - 18}" y2="{y2}" '
                    f'stroke="{LINE}" stroke-opacity="{opacity}"/>'
                )

    svg_connect(x_input, shared_y, x_gru, shared_y, 0.22)
    svg_connect(x_gru, shared_y, x_feature, shared_y, 0.19)
    svg_connect(x_feature, shared_y, x_hidden_1, actor_y, 0.15)
    svg_connect(x_hidden_1, actor_y, x_hidden_2, actor_y, 0.18)
    svg_connect(x_hidden_2, actor_y, x_output, actor_output_y, 0.15)
    svg_connect(x_feature, shared_y, x_hidden_1, critic_y, 0.15)
    svg_connect(x_hidden_1, critic_y, x_hidden_2, critic_y, 0.18)
    svg_connect(x_hidden_2, critic_y, x_output, critic_output_y, 0.22)
    svg.append('</g><g font-family="Microsoft YaHei, Segoe UI, sans-serif">')

    svg.append(text_svg(WIDTH / 2, 54, "PPO Actor–Critic 网络结构", size=48, weight=500))
    svg.append(text_svg(WIDTH / 2, 108, "默认障碍场 GRU 配置  ·  历史长度 20  ·  单帧观测 28 维", size=26, fill=MUTED))
    svg.append(f'<line x1="1420" y1="155" x2="2040" y2="155" stroke="{ACTOR_COLOR}" stroke-width="3"/>')
    svg.append(f'<line x1="2390" y1="155" x2="3010" y2="155" stroke="{ACTOR_COLOR}" stroke-width="3"/>')
    svg.append(text_svg(2215, 155, "Actor 策略分支", size=30, fill=ACTOR_COLOR, weight=500))
    svg.append(f'<line x1="1420" y1="1080" x2="2030" y2="1080" stroke="{CRITIC_COLOR}" stroke-width="3"/>')
    svg.append(f'<line x1="2400" y1="1080" x2="3010" y2="1080" stroke="{CRITIC_COLOR}" stroke-width="3"/>')
    svg.append(text_svg(2215, 1080, "Critic 价值分支", size=30, fill=CRITIC_COLOR, weight=500))

    for x, title, shape, detail in shared_headers:
        svg.append(text_svg(x, 355, title, size=28, weight=500))
        svg.append(text_svg(x, 394, shape, size=23, fill=MUTED))
        svg.append(text_svg(x, 429, detail, size=20, fill=MUTED))
    for x, title, shape, detail in actor_headers:
        svg.append(text_svg(x, 202, title, size=27, fill=ACTOR_COLOR, weight=500))
        svg.append(text_svg(x, 238, shape, size=22, fill=MUTED))
        svg.append(text_svg(x, 268, detail, size=19, fill=MUTED))
    for x, title, shape, detail in critic_headers:
        svg.append(text_svg(x, 1122, title, size=27, fill=CRITIC_COLOR, weight=500))
        svg.append(text_svg(x, 1158, shape, size=22, fill=MUTED))
        svg.append(text_svg(x, 1188, detail, size=19, fill=MUTED))

    for y in shared_y:
        svg.append(node_svg(x_input, y, INPUT_COLOR))
        svg.append(node_svg(x_gru, y, GRU_COLOR))
        svg.append(node_svg(x_feature, y, FEATURE_COLOR))
    for y in actor_y:
        svg.append(node_svg(x_hidden_1, y, ACTOR_COLOR))
        svg.append(node_svg(x_hidden_2, y, ACTOR_COLOR))
    for y in critic_y:
        svg.append(node_svg(x_hidden_1, y, CRITIC_COLOR))
        svg.append(node_svg(x_hidden_2, y, CRITIC_COLOR))
    for x, ys in (
        (x_input, (910, 960, 1010)),
        (x_gru, (910, 960, 1010)),
        (x_feature, (910, 960, 1010)),
        (x_hidden_1, (626, 656, 686)),
        (x_hidden_2, (626, 656, 686)),
        (x_hidden_1, (1470, 1500, 1530)),
        (x_hidden_2, (1470, 1500, 1530)),
    ):
        for y in ys:
            svg.append(text_svg(x, y, "·", size=38, fill=MUTED))

    for y, color, index in zip(actor_output_y, output_colors, output_indices):
        svg.append(node_svg(x_output, y, color, radius=14))
        svg.append(text_svg(x_output + 34, y, str(index), size=17, fill=MUTED, anchor="start"))
    svg.append(node_svg(x_output, critic_output_y[0], CRITIC_COLOR, radius=25))
    svg.append(text_svg(x_output, critic_output_y[0], "V", size=22, fill=CRITIC_COLOR, weight=500))

    svg.append(f'<path d="M220 500 L196 518 L196 1572 L220 1590" fill="none" stroke="{TEXT}" stroke-width="5"/>')
    svg.append(text_svg(110, 995, "28", size=52, weight=500))
    svg.append(text_svg(110, 1048, "维／帧", size=23, fill=MUTED))
    svg.append(text_svg(110, 1091, "× 20 帧", size=23, fill=INPUT_COLOR, weight=500))

    for y1, y2, color, label, dim in action_groups:
        center = (y1 + y2) / 2
        svg.append(f'<path d="M2819 {y1} L2795 {y1 + 18} L2795 {y2 - 18} L2819 {y2}" fill="none" stroke="{color}" stroke-width="5"/>')
        svg.append(text_svg(2842, center - 17, label, size=24, fill=color, anchor="start", weight=500))
        svg.append(text_svg(2842, center + 20, dim, size=19, fill=MUTED, anchor="start"))
    svg.append(text_svg(3030, 620, "三组分别形成", size=22, fill=MUTED, anchor="start"))
    svg.append(text_svg(3030, 655, "Categorical 分布", size=22, fill=MUTED, anchor="start"))

    svg.append(text_svg(x_gru, 1605, "沿 20 帧递推，取最后隐藏状态", size=21, fill=GRU_COLOR))
    svg.append(text_svg(x_gru, 1641, "GRU：18,048 参数", size=20, fill=MUTED))
    svg.append(text_svg(x_feature, 1605, "共享给 Actor 与 Critic", size=21, fill=FEATURE_COLOR))
    svg.append(text_svg(x_feature, 1641, "投影层：4,160 参数", size=20, fill=MUTED))
    svg.append(text_svg(x_output + 65, 1512, "V(S)", size=25, fill=CRITIC_COLOR, anchor="start", weight=500))
    svg.append(text_svg(x_output + 65, 1548, "每个样本一个价值标量", size=20, fill=MUTED, anchor="start"))

    svg.append(f'<line x1="200" y1="1810" x2="3200" y2="1810" stroke="{STRUCTURE}" stroke-width="2"/>')
    svg.append(text_svg(WIDTH / 2, 1855, "共享编码器 22,208 参数  ·  Actor 40,346 参数  ·  Critic 36,321 参数  ·  总计 98,875 参数", size=25, fill=MUTED))
    svg.append(text_svg(WIDTH / 2, 1900, "B 为批量大小；无障碍配置仅把输入改为 B × 20 × 16（即 B × 320），其余网络层保持不变", size=22, fill=MUTED))
    svg.append("</g></svg>")
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    build()
