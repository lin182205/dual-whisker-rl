"""Repeated sector scan for checking servo motion and MQ-3 responses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.hardware.hardware_logger import save_samples_csv
from dual_whisker_rl.hardware.serial_client import DualWhiskerSerialClient
from dual_whisker_rl.paths import resolve_path_args
from dual_whisker_rl.hardware.serial_client import HardwareSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--delay-s", type=float, default=0.2)
    parser.add_argument("--csv-path", type=Path, default=Path("results/hardware/sector_scan.csv"))
    parser.add_argument("--figure-path", type=Path, default=Path("results/hardware/sector_scan.png"))
    return resolve_path_args(parser.parse_args(), "config", "csv_path", "figure_path")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    serial_cfg = config["serial"]
    sector_count = int(config["whisker"]["sector_count"])
    port = args.port or str(serial_cfg["port"])

    samples: list[HardwareSample] = []
    with DualWhiskerSerialClient(
        port=port,
        baudrate=int(serial_cfg["baudrate"]),
        timeout_s=float(serial_cfg["timeout_s"]),
    ) as client:
        for cycle in range(args.cycles):
            for sector in range(sector_count):
                sample = client.step(sector, sector)
                samples.append(sample)
                print(
                    f"cycle={cycle} sector={sector} left_adc={sample.left_adc} "
                    f"right_adc={sample.right_adc}"
                )
                time.sleep(args.delay_s)

    save_samples_csv(samples, args.csv_path)
    save_sector_scan_plot(samples, sector_count, args.figure_path)
    print(f"saved_csv={args.csv_path}")
    print(f"saved_figure={args.figure_path}")


def save_sector_scan_plot(
    samples: list[HardwareSample],
    sector_count: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for sector in range(sector_count):
        sector_samples = [sample for sample in samples if sample.left_sector == sector]
        if not sector_samples:
            continue
        left_values = [sample.left_adc for sample in sector_samples]
        right_values = [sample.right_adc for sample in sector_samples]
        axes[0].scatter([sector] * len(left_values), left_values, color="#4c78a8", alpha=0.7)
        axes[1].scatter([sector] * len(right_values), right_values, color="#f58518", alpha=0.7)

    axes[0].set_ylabel("left ADC")
    axes[1].set_ylabel("right ADC")
    axes[1].set_xlabel("sector")
    axes[0].set_title("Repeated sector scan")
    axes[1].set_xticks(range(sector_count))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
