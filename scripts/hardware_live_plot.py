"""Live plot STM32 dual-whisker MQ-3 sensor responses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import queue
import sys
import threading
import time

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.hardware.serial_client import DualWhiskerSerialClient
from dual_whisker_rl.hardware.serial_client import HardwareSample
from dual_whisker_rl.paths import resolve_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hardware.yaml"))
    parser.add_argument("--port", type=str, default= "COM6")
    parser.add_argument("--left-sector", type=int, default=5)
    parser.add_argument("--right-sector", type=int, default=5)
    parser.add_argument("--interval-s", type=float, default=0.0)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--max-s", type=float, default=None)
    parser.add_argument("--csv-path", type=Path, default=Path("results/hardware/live_plot.csv"))
    parser.add_argument("--adc-max", type=float, default=4095.0)
    parser.add_argument("--voltage", action="store_true")
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return resolve_path_args(parser.parse_args(), "config", "csv_path")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adc_to_voltage(adc: int, adc_max: float, vref: float) -> float:
    return float(adc) / adc_max * vref


def sample_worker(
    client: DualWhiskerSerialClient,
    left_sector: int,
    right_sector: int,
    interval_s: float,
    samples_queue: queue.Queue[HardwareSample | BaseException],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            sample = client.step(left_sector, right_sector)
        except BaseException as exc:
            samples_queue.put(exc)
            stop_event.set()
            return

        samples_queue.put(sample)
        if interval_s > 0.0 and stop_event.wait(interval_s):
            return


def save_live_samples_csv(
    samples: list[HardwareSample],
    path: Path,
    t0: float,
    adc_max: float,
    vref: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "t_rel_s",
                "t_pc_s",
                "left_sector",
                "right_sector",
                "left_adc",
                "right_adc",
                "left_voltage",
                "right_voltage",
            ],
        )
        writer.writeheader()
        for idx, sample in enumerate(samples):
            writer.writerow(
                {
                    "step": idx,
                    "t_rel_s": f"{sample.t_pc_s - t0:.6f}",
                    "t_pc_s": f"{sample.t_pc_s:.6f}",
                    "left_sector": sample.left_sector,
                    "right_sector": sample.right_sector,
                    "left_adc": sample.left_adc,
                    "right_adc": sample.right_adc,
                    "left_voltage": f"{adc_to_voltage(sample.left_adc, adc_max, vref):.6f}",
                    "right_voltage": f"{adc_to_voltage(sample.right_adc, adc_max, vref):.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    serial_cfg = config["serial"]
    sector_count = int(config["whisker"]["sector_count"])
    vref = float(config.get("stm32", {}).get("adc_max_voltage", 3.3))
    port = args.port or str(serial_cfg["port"])

    if not 0 <= args.left_sector < sector_count:
        raise ValueError(f"--left-sector must be in [0, {sector_count - 1}]")
    if not 0 <= args.right_sector < sector_count:
        raise ValueError(f"--right-sector must be in [0, {sector_count - 1}]")

    samples: list[HardwareSample] = []
    samples_queue: queue.Queue[HardwareSample | BaseException] = queue.Queue()
    stop_event = threading.Event()

    xs: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    t0: float | None = None
    start_wall_s = time.time()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    left_line, = ax.plot([], [], color="#4c78a8", linewidth=2.0, label="left MQ-3")
    right_line, = ax.plot([], [], color="#f58518", linewidth=2.0, label="right MQ-3")
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("voltage (V)" if args.voltage else "ADC value")
    ax.set_title(
        "Live MQ-3 response "
        f"(left sector {args.left_sector}, right sector {args.right_sector})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    status_text = ax.text(
        0.01,
        0.98,
        "waiting for samples...",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    def request_stop(_event) -> None:
        stop_event.set()

    fig.canvas.mpl_connect("close_event", request_stop)

    with DualWhiskerSerialClient(
        port=port,
        baudrate=int(serial_cfg["baudrate"]),
        timeout_s=float(serial_cfg["timeout_s"]),
    ) as client:
        worker = threading.Thread(
            target=sample_worker,
            args=(
                client,
                args.left_sector,
                args.right_sector,
                args.interval_s,
                samples_queue,
                stop_event,
            ),
            daemon=True,
        )
        worker.start()

        try:
            while not stop_event.is_set():
                while True:
                    try:
                        item = samples_queue.get_nowait()
                    except queue.Empty:
                        break

                    if isinstance(item, BaseException):
                        raise item

                    sample = item
                    if t0 is None:
                        t0 = sample.t_pc_s

                    t_rel_s = sample.t_pc_s - t0
                    samples.append(sample)
                    xs.append(t_rel_s)

                    if args.voltage:
                        left_y = adc_to_voltage(sample.left_adc, args.adc_max, vref)
                        right_y = adc_to_voltage(sample.right_adc, args.adc_max, vref)
                    else:
                        left_y = float(sample.left_adc)
                        right_y = float(sample.right_adc)

                    left_values.append(left_y)
                    right_values.append(right_y)

                    if not args.quiet:
                        print(
                            f"t={t_rel_s:8.3f}s "
                            f"left={sample.left_adc:4d} right={sample.right_adc:4d}"
                        )

                if xs:
                    left_line.set_data(xs, left_values)
                    right_line.set_data(xs, right_values)

                    x_end = max(args.window_s, xs[-1])
                    x_start = max(0.0, x_end - args.window_s)
                    ax.set_xlim(x_start, x_end)

                    if args.y_min is not None and args.y_max is not None:
                        ax.set_ylim(args.y_min, args.y_max)
                    else:
                        visible = [
                            y
                            for x, y in zip(xs, left_values)
                            if x_start <= x <= x_end
                        ]
                        visible.extend(
                            y for x, y in zip(xs, right_values) if x_start <= x <= x_end
                        )
                        if visible:
                            y_min = min(visible)
                            y_max = max(visible)
                            margin = max(1.0, (y_max - y_min) * 0.15)
                            ax.set_ylim(y_min - margin, y_max + margin)

                    status_text.set_text(
                        f"samples={len(samples)}  "
                        f"last left={samples[-1].left_adc}  "
                        f"last right={samples[-1].right_adc}"
                    )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.03)

                if args.max_s is not None and time.time() - start_wall_s >= args.max_s:
                    stop_event.set()
                if not plt.fignum_exists(fig.number):
                    stop_event.set()

        except KeyboardInterrupt:
            stop_event.set()
        finally:
            stop_event.set()
            worker.join(timeout=2.0)

    if samples and not args.no_save and t0 is not None:
        save_live_samples_csv(samples, args.csv_path, t0, args.adc_max, vref)
        print(f"saved_csv={args.csv_path}")
    elif not samples:
        print("no samples captured")


if __name__ == "__main__":
    main()
