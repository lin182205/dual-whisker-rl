"""Run a minimal STM32 dual-whisker hardware smoke test."""

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
from dual_whisker_rl.hardware.serial_client import HardwareSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hardware.yaml")
    parser.add_argument("--port", type=str, default= "COM6")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--delay-s", type=float, default=0.2)
    parser.add_argument("--csv-path", type=Path, default=ROOT / "results" / "hardware" / "smoke_test.csv")
    parser.add_argument("--figure-path", type=Path, default=ROOT / "results" / "hardware" / "smoke_test.png")
    return parser.parse_args()


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
        for _ in range(args.cycles):
            for sector in range(sector_count):
                sample = client.step(sector, sector)
                samples.append(sample)
                print(
                    f"sector={sector} left_adc={sample.left_adc} "
                    f"right_adc={sample.right_adc}"
                )
                time.sleep(args.delay_s)

    save_samples_csv(samples, args.csv_path)
    save_adc_plot(samples, args.figure_path)
    print(f"saved_csv={args.csv_path}")
    print(f"saved_figure={args.figure_path}")


def save_adc_plot(samples: list[HardwareSample], path: Path) -> None:
    steps = list(range(len(samples)))
    left = [sample.left_adc for sample in samples]
    right = [sample.right_adc for sample in samples]
    labels = [sample.left_sector for sample in samples]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, left, marker="o", color="#4c78a8", label="left MQ-3 ADC")
    ax.plot(steps, right, marker="o", color="#f58518", label="right MQ-3 ADC")
    ax.set_xlabel("step")
    ax.set_ylabel("ADC value")
    ax.set_xticks(steps)
    ax.set_xticklabels(labels)
    ax.set_title("Hardware smoke test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
