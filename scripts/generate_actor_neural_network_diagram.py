"""生成 Actor 专属神经元连接结构图（SVG 与 PNG）。"""

from __future__ import annotations

from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2400
HEIGHT = 1500
BACKGROUND = "#0b0f14"
TEXT = "#f1f5f9"
MUTED = "#aeb8c6"
LINE = "#aeb8c6"
CYAN = "#58c7df"
TEAL = "#55d6be"
BLUE = "#70a7ff"
GREEN = "#79d98c"
ORANGE = "#f2aa63"

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
PNG_PATH = FIGURE_DIR / "algorithm-actor-only-neural-network.png"
SVG_PATH = FIGURE_DIR / "algorithm-actor-only-neural-network.svg"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def draw_text(
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


def node(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, radius: int = 17) -> None:
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=BACKGROUND,
        outline=color,
        width=5,
    )


def bracket(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int, color: str) -> None:
    draw.line((x + 24, y1, x, y1 + 18, x, y2 - 18, x + 24, y2), fill=color, width=5)


def svg_text(
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


def svg_node(x: int, y: int, color: str, radius: int = 17) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{BACKGROUND}" '
        f'stroke="{color}" stroke-width="5"/>'
    )


def build() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_input, x_hidden_1, x_hidden_2, x_output = 330, 850, 1370, 1850
    input_y = [230 + 60 * i for i in range(6)] + [950 + 60 * i for i in range(6)]
    hidden_1_y = [220 + 55 * i for i in range(8)] + [870 + 55 * i for i in range(8)]
    hidden_2_y = hidden_1_y.copy()
    output_y = [220 + 40 * i for i in range(26)]

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    connections = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    connection_draw = ImageDraw.Draw(connections)

    for y0 in input_y:
        for y1 in hidden_1_y:
            connection_draw.line((x_input + 18, y0, x_hidden_1 - 18, y1), fill=(174, 184, 198, 55), width=1)
    for y0 in hidden_1_y:
        for y1 in hidden_2_y:
            connection_draw.line((x_hidden_1 + 18, y0, x_hidden_2 - 18, y1), fill=(174, 184, 198, 48), width=1)
    for y0 in hidden_2_y:
        for y1 in output_y:
            connection_draw.line((x_hidden_2 + 18, y0, x_output - 18, y1), fill=(174, 184, 198, 44), width=1)
    image = Image.alpha_composite(image.convert("RGBA"), connections).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw_text(draw, (WIDTH / 2, 52), "Actor 网络结构", size=46, bold=True)
    draw_text(draw, (WIDTH / 2, 101), "64 → 160 → 160 → 26 logits", size=28, fill=MUTED)

    headers = [
        (x_input, "Actor 输入", "B × 64"),
        (x_hidden_1, "隐藏层 1", "B × 160"),
        (x_hidden_2, "隐藏层 2", "B × 160"),
        (x_output, "动作输出层", "B × 26"),
    ]
    for x, title, shape in headers:
        draw_text(draw, (x, 137), title, size=29, bold=True)
        draw_text(draw, (x, 168), shape, size=21, fill=MUTED)

    for y in input_y:
        node(draw, x_input, y, CYAN)
    for y in hidden_1_y:
        node(draw, x_hidden_1, y, TEAL)
    for y in hidden_2_y:
        node(draw, x_hidden_2, y, BLUE)

    for x in (x_input, x_hidden_1, x_hidden_2):
        draw_text(draw, (x, 700), "·", size=42, fill=MUTED)
        draw_text(draw, (x, 735), "·", size=42, fill=MUTED)
        draw_text(draw, (x, 770), "·", size=42, fill=MUTED)

    output_colors = [CYAN] * 6 + [GREEN] * 10 + [ORANGE] * 10
    output_indices = list(range(6)) + list(range(10)) + list(range(10))
    for y, color, index in zip(output_y, output_colors, output_indices):
        node(draw, x_output, y, color)
        draw_text(draw, (x_output + 38, y), str(index), size=20, fill=MUTED, anchor="lm")

    bracket(draw, 208, 205, 1295, TEXT)
    draw_text(draw, (128, 750), "64", size=54, bold=True)
    draw_text(draw, (128, 807), "维特征", size=24, fill=MUTED)

    groups = [
        (200, 440, CYAN, "移动动作", "6 个 logits"),
        (440, 840, GREEN, "左触须扇区", "10 个 logits"),
        (840, 1240, ORANGE, "右触须扇区", "10 个 logits"),
    ]
    for y1, y2, color, label, dim in groups:
        bracket(draw, 1970, y1, y2, color)
        center = (y1 + y2) / 2
        draw_text(draw, (2020, center - 18), label, size=27, fill=color, anchor="lm", bold=True)
        draw_text(draw, (2020, center + 22), dim, size=21, fill=MUTED, anchor="lm")

    layer_notes = [
        (x_input, "特征向量", "批量维为 B"),
        (x_hidden_1, "Linear 64 → 160", "Tanh  ·  10,400 参数"),
        (x_hidden_2, "Linear 160 → 160", "Tanh  ·  25,760 参数"),
        (x_output, "Linear 160 → 26", "无激活  ·  4,186 参数"),
    ]
    for x, first, second in layer_notes:
        draw_text(draw, (x, 1340), first, size=26, bold=True)
        draw_text(draw, (x, 1382), second, size=21, fill=MUTED)

    draw.line((220, 1422, 2180, 1422), fill="#334155", width=2)
    draw_text(
        draw,
        (WIDTH / 2, 1460),
        "Actor 专属可训练参数：40,346  ·  所有相邻层均为全连接",
        size=24,
        fill=MUTED,
    )
    image.save(PNG_PATH, format="PNG", optimize=True)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<title>Actor 网络结构</title>",
        "<desc>Actor 输入为六十四维特征，经过两个一百六十维 Tanh 全连接隐藏层，输出二十六个动作 logits，并分为三组，数量分别是六、十和十。</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        '<g fill="none" stroke-linecap="round">',
    ]
    for y0 in input_y:
        for y1 in hidden_1_y:
            svg.append(f'<line x1="{x_input + 18}" y1="{y0}" x2="{x_hidden_1 - 18}" y2="{y1}" stroke="{LINE}" stroke-opacity="0.22"/>')
    for y0 in hidden_1_y:
        for y1 in hidden_2_y:
            svg.append(f'<line x1="{x_hidden_1 + 18}" y1="{y0}" x2="{x_hidden_2 - 18}" y2="{y1}" stroke="{LINE}" stroke-opacity="0.19"/>')
    for y0 in hidden_2_y:
        for y1 in output_y:
            svg.append(f'<line x1="{x_hidden_2 + 18}" y1="{y0}" x2="{x_output - 18}" y2="{y1}" stroke="{LINE}" stroke-opacity="0.17"/>')
    svg.append('</g><g font-family="Microsoft YaHei, Segoe UI, sans-serif">')
    svg.append(svg_text(WIDTH / 2, 52, "Actor 网络结构", size=46, weight=500))
    svg.append(svg_text(WIDTH / 2, 101, "64 → 160 → 160 → 26 logits", size=28, fill=MUTED))
    for x, title, shape in headers:
        svg.append(svg_text(x, 137, title, size=29, weight=500))
        svg.append(svg_text(x, 168, shape, size=21, fill=MUTED))
    for y in input_y:
        svg.append(svg_node(x_input, y, CYAN))
    for y in hidden_1_y:
        svg.append(svg_node(x_hidden_1, y, TEAL))
    for y in hidden_2_y:
        svg.append(svg_node(x_hidden_2, y, BLUE))
    for x in (x_input, x_hidden_1, x_hidden_2):
        for y in (700, 735, 770):
            svg.append(svg_text(x, y, "·", size=42, fill=MUTED))
    for y, color, index in zip(output_y, output_colors, output_indices):
        svg.append(svg_node(x_output, y, color))
        svg.append(svg_text(x_output + 38, y, str(index), size=20, fill=MUTED, anchor="start"))
    svg.append(f'<path d="M232 205 L208 223 L208 1277 L232 1295" fill="none" stroke="{TEXT}" stroke-width="5"/>')
    svg.append(svg_text(128, 750, "64", size=54, weight=500))
    svg.append(svg_text(128, 807, "维特征", size=24, fill=MUTED))
    for y1, y2, color, label, dim in groups:
        center = (y1 + y2) / 2
        svg.append(f'<path d="M1994 {y1} L1970 {y1 + 18} L1970 {y2 - 18} L1994 {y2}" fill="none" stroke="{color}" stroke-width="5"/>')
        svg.append(svg_text(2020, center - 18, label, size=27, fill=color, anchor="start", weight=500))
        svg.append(svg_text(2020, center + 22, dim, size=21, fill=MUTED, anchor="start"))
    for x, first, second in layer_notes:
        svg.append(svg_text(x, 1340, first, size=26, weight=500))
        svg.append(svg_text(x, 1382, second, size=21, fill=MUTED))
    svg.append('<line x1="220" y1="1422" x2="2180" y2="1422" stroke="#334155" stroke-width="2"/>')
    svg.append(svg_text(WIDTH / 2, 1460, "Actor 专属可训练参数：40,346  ·  所有相邻层均为全连接", size=24, fill=MUTED))
    svg.append("</g></svg>")
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")

    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    build()
