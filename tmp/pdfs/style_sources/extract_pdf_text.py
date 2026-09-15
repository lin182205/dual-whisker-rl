from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def source_argument(value: str) -> tuple[str, Path]:
    """解析 ``名称=PDF路径``，避免把本机文献目录写入脚本。"""
    slug, separator, path_text = value.partition("=")
    if not separator or not slug.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("文献参数格式必须为 名称=PDF路径")
    return slug.strip(), Path(path_text.strip()).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将指定 PDF 按页提取为文本")
    parser.add_argument("sources", nargs="+", type=source_argument, metavar="名称=PDF路径")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = dict(args.sources)
    if len(sources) != len(args.sources):
        raise ValueError("文献名称不能重复")
    for slug, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"PDF 文件不存在: {source}")
        reader = PdfReader(str(source))
        chunks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            chunks.append(f"\n\n===== PAGE {page_number} =====\n\n{text}")
        target = output_dir / f"{slug}.txt"
        target.write_text("".join(chunks), encoding="utf-8")
        print(f"{slug}: {len(reader.pages)} pages, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
