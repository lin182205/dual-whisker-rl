from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


SOURCES = {
    "recurrent": Path(r"C:\Users\86189\Desktop\嗅觉鸡\DRL论文\Robotic_Odor_Source_Localization_via_End-to-End_Recurrent_Deep_Reinforcement_Learning.pdf"),
    "dqn_comparative": Path(r"C:\Users\86189\Desktop\嗅觉鸡\DRL论文\A Deep Q-Network for robotic odorgas source localization Modeling, measurement and comparative study.pdf"),
    "dqn_modeling": Path(r"C:\Users\86189\Desktop\嗅觉鸡\DRL论文\A Deep Q-Network for robotic odorgas source localization Modeling, .pdf"),
}


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    for slug, source in SOURCES.items():
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
