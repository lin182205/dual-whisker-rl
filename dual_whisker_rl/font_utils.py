"""跨平台中文字体发现工具。"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont

from dual_whisker_rl.paths import project_path


FONT_REGULAR_ENV = "DUAL_WHISKER_FONT_REGULAR"
FONT_BOLD_ENV = "DUAL_WHISKER_FONT_BOLD"
CHINESE_FONT_FAMILIES = (
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "SimHei",
)


@lru_cache(maxsize=2)
def resolve_chinese_font(*, bold: bool = False) -> Path:
    """返回可用中文字体；允许通过环境变量覆盖系统自动发现结果。"""
    configured = os.environ.get(FONT_BOLD_ENV if bold else FONT_REGULAR_ENV)
    if bold and not configured:
        configured = os.environ.get(FONT_REGULAR_ENV)
    if configured:
        path = project_path(configured)
        if path is not None and path.is_file():
            return path
        raise FileNotFoundError(f"字体环境变量指向的文件不存在: {configured}")

    properties = FontProperties(
        family=list(CHINESE_FONT_FAMILIES),
        weight="bold" if bold else "normal",
    )
    try:
        path = Path(findfont(properties, fallback_to_default=False))
    except ValueError as exc:
        raise FileNotFoundError(
            "未发现可用中文字体。Linux 可安装 fonts-noto-cjk，或设置 "
            f"{FONT_REGULAR_ENV}/{FONT_BOLD_ENV}。"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"字体发现结果不存在: {path}")
    return path
