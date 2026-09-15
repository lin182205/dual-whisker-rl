from __future__ import annotations

import importlib.util
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "active-olfaction-pape"
EXPORT_SCRIPT = REPO / "scripts" / "export_paper_docx.py"
OUTPUT = REPO / "output" / "docx" / "active_olfaction_paper_reference_style_revised.docx"


def main() -> None:
    spec = importlib.util.spec_from_file_location("paper_docx_export", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载导出脚本：{EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = OUTPUT
    module.main()


if __name__ == "__main__":
    main()
