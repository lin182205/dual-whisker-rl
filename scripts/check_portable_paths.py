"""检查运行代码和配置中是否出现机器相关的绝对路径字面量。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKED_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".ps1",
    ".sh",
    ".bat",
    ".code-workspace",
}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    ".claude",
    ".codex",
    ".pytest_cache",
    "results",
    "build",
    "dist",
    "__pycache__",
    "node_modules",
}
PATTERNS = {
    "Windows 盘符路径": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "Linux/macOS 用户或挂载路径": re.compile(
        r'''["']/(?:home|Users|mnt|root|tmp|opt|var(?:/tmp)?)/'''
    ),
    "UNC 网络路径": re.compile(
        r'''(?:[rubf]{0,2})?["']\\\\[A-Za-z0-9._-]+[\\/][A-Za-z0-9$._-]+''',
        re.IGNORECASE,
    ),
}


def configure_output_encoding() -> None:
    """重定向日志统一写 UTF-8；交互式 Windows 终端沿用本地编码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and not stream.isatty():
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="待检查仓库；默认当前项目根目录")
    return parser.parse_args()


def iter_source_files(root: Path):
    checker = Path(__file__).resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKED_SUFFIXES:
            continue
        if any(part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        # 审计器自身包含用于识别绝对路径的正则表达式，不参与扫描。
        if path.resolve() == checker:
            continue
        yield path


def main() -> None:
    configure_output_encoding()
    args = parse_args()
    root = args.root.expanduser().resolve()
    findings: list[tuple[Path, int, str, str]] = []
    for path in iter_source_files(root):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((path.relative_to(root), line_number, label, line.strip()))

    if findings:
        print("发现机器相关的绝对路径：")
        for path, line_number, label, line in findings:
            print(f"- {path.as_posix()}:{line_number} [{label}] {line}")
        raise SystemExit(1)
    print(f"路径检查通过：未发现机器相关的绝对路径；检查根目录={root}")


if __name__ == "__main__":
    main()
