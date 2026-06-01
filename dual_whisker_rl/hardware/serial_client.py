"""Serial client for the STM32 dual-whisker smoke-test firmware."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

import serial


@dataclass
class HardwareSample:
    left_sector: int
    right_sector: int
    left_adc: int
    right_adc: int
    t_pc_s: float

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> "HardwareSample":
        return cls(
            left_sector=int(response["left_sector"]),
            right_sector=int(response["right_sector"]),
            left_adc=int(response["left_adc"]),
            right_adc=int(response["right_adc"]),
            t_pc_s=time.time(),
        )


class DualWhiskerSerialClient:
    """Small text-protocol client for `STEP left right` commands."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout_s: float = 2.0,
    ) -> None:
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout_s,
            write_timeout=timeout_s,
        )

    def close(self) -> None:
        self.serial.close()

    def step(self, left_sector: int, right_sector: int) -> HardwareSample:
        command = f"STEP {int(left_sector)} {int(right_sector)}\n"
        self.serial.write(command.encode("ascii"))
        self.serial.flush()

        while True:
            line = self.serial.readline().decode("utf-8", errors="replace").strip()
            if not line:
                raise TimeoutError("No response from STM32")

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid STM32 response: {line!r}") from exc

            if "error" in response:
                raise RuntimeError(f"STM32 error response: {response['error']}")
            if {"left_adc", "right_adc", "left_sector", "right_sector"} <= response.keys():
                return HardwareSample.from_response(response)

    def __enter__(self) -> "DualWhiskerSerialClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
