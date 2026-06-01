"""Logging helpers for hardware smoke-test samples."""

from __future__ import annotations

import csv
from pathlib import Path

from dual_whisker_rl.hardware.serial_client import HardwareSample


FIELDNAMES = [
    "step",
    "t_pc_s",
    "left_sector",
    "right_sector",
    "left_adc",
    "right_adc",
]


def save_samples_csv(samples: list[HardwareSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            writer.writerow(
                {
                    "step": idx,
                    "t_pc_s": f"{sample.t_pc_s:.6f}",
                    "left_sector": sample.left_sector,
                    "right_sector": sample.right_sector,
                    "left_adc": sample.left_adc,
                    "right_adc": sample.right_adc,
                }
            )
