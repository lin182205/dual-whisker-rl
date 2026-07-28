"""项目内路径解析工具。

命令行统一使用相对仓库根目录的路径，避免把本机盘符或云服务器目录写入
默认参数和运行元数据。用户显式提供绝对路径时仍原样保留，便于使用挂载盘。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path | None) -> Path | None:
    """把相对路径按仓库根目录解析；绝对路径保持不变。"""
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_path_args(
    args: argparse.Namespace,
    *attribute_names: str,
) -> argparse.Namespace:
    """原地解析 Namespace 中指定的可选路径字段。"""
    for name in attribute_names:
        value: Any = getattr(args, name)
        setattr(args, name, project_path(value))
    return args


def portable_path(path: str | Path | None) -> str | None:
    """用于日志/metadata：仓库内路径写成 POSIX 风格相对路径。"""
    resolved = project_path(path)
    if resolved is None:
        return None
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        # 显式指定的仓库外挂载路径无法表示成项目相对路径，保留其可用形式。
        return resolved.as_posix()
