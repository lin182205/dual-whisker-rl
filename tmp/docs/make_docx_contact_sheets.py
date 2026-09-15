from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent / "reference_style_render"
PAGES = sorted(ROOT.glob("page-*.png"))
OUT = ROOT / "contact_sheets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sheet_index in range(0, len(PAGES), 4):
        group = PAGES[sheet_index : sheet_index + 4]
        opened = [Image.open(path).convert("RGB") for path in group]
        target_width = 850
        scaled = []
        for image in opened:
            height = round(image.height * target_width / image.width)
            scaled.append(image.resize((target_width, height)))
        cell_height = max(image.height for image in scaled) + 40
        canvas = Image.new("RGB", (target_width * 2 + 30, cell_height * 2 + 30), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, (image, path) in enumerate(zip(scaled, group)):
            row, col = divmod(offset, 2)
            x = col * (target_width + 30)
            y = row * cell_height + 28
            canvas.paste(image, (x, y))
            draw.text((x + 8, row * cell_height + 6), path.stem, fill="black")
        first_page = sheet_index + 1
        last_page = sheet_index + len(group)
        canvas.save(OUT / f"pages_{first_page:02d}_{last_page:02d}.png")
    print(f"created {len(list(OUT.glob('*.png')))} contact sheets")


if __name__ == "__main__":
    main()
