"""仿真评估的终端进度；估时只使用本次新运行场景的耗时。"""
from __future__ import annotations

import math
import sys
import time

# 仅允许这一次已审查的显示升级兼容原封存，不放宽其他代码或配置变化。
PRE_ETA_RUNNER_SHA256 = "4236f4240a76ef99d15ea1e615c643fab3db5af7efb7292b90a27e30807f63c0"
ETA_RUNNER_SHA256 = "aebc4866f1c800ffd246fac97834fa750bd15fa2e93412fb88043453a2a1985d"
RUNNER_PATH = "scripts/run_red_experiments.py"


def reporting_compatible_payload(payload, saved):
    """只接受已知运行器版本的显示升级，完整保留原数据的签名。"""
    from dual_whisker_rl.experiment_protocol import digest

    hashes = payload.get("code_sha256", {})
    if hashes.get(RUNNER_PATH) != ETA_RUNNER_SHA256:
        return payload
    if saved.get("code_sha256", {}).get(RUNNER_PATH) != PRE_ETA_RUNNER_SHA256:
        return payload
    previous = {**payload, "code_sha256": {**hashes, RUNNER_PATH: PRE_ETA_RUNNER_SHA256}}
    if saved == {**previous, "signature": digest(previous)}:
        return previous
    return payload


def configure_output_encoding() -> None:
    """重定向日志使用UTF-8；交互式Windows终端保持其本地编码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and not stream.isatty():
            reconfigure(encoding="utf-8", errors="replace")


def duration_text(seconds: float, *, round_up: bool = True) -> str:
    """时长按时:分:秒显示，未结束时避免舍入成零秒。"""
    value = math.ceil(max(0.0, seconds)) if round_up else int(max(0.0, seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EvaluationProgress:
    """已校验缓存先计入进度，但不计入速度样本；无需后台线程。"""

    def __init__(self, total: int, *, cached: int = 0, every: int = 20,
                 interval_s: float = 10.0, clock=None, printer=None):
        if total < 0 or not 0 <= cached <= total or every < 1 or interval_s <= 0:
            raise ValueError("进度数量或刷新间隔无效")
        self.total = total
        self.cached = cached
        self.new_completed = 0
        self.measured_s = 0.0
        self.every = every
        self.interval_s = interval_s
        self.clock = clock or time.perf_counter
        self.printer = printer or print
        self.started = self.clock()
        self.last_reported_at = self.started
        self.last_reported_new = -1

    def finish_episode(self, duration_s: float) -> None:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("场景耗时必须是有限非负数")
        if self.cached + self.new_completed >= self.total:
            raise ValueError("已完成全部场景")
        self.new_completed += 1
        self.measured_s += duration_s

    def snapshot(self):
        completed = self.cached + self.new_completed
        remaining = self.total - completed
        eta_s = (0.0 if remaining == 0 else
                 self.measured_s / self.new_completed * remaining if self.new_completed else None)
        return dict(completed=completed, total=self.total, cached=self.cached,
                    new_completed=self.new_completed,
                    elapsed_s=max(0.0, self.clock() - self.started), eta_s=eta_s)

    def report(self, label: str, *, force: bool = False) -> bool:
        state = self.snapshot()
        now = self.clock()
        changed = self.new_completed != self.last_reported_new
        due = (self.last_reported_new < 0 or
               (self.new_completed == 1 and self.last_reported_new < 1) or
               self.new_completed - self.last_reported_new >= self.every or
               now - self.last_reported_at >= self.interval_s or
               state["completed"] == self.total)
        if not force and not (changed and due):
            return False
        eta = "等待首个新场景完成" if state["eta_s"] is None else duration_text(state["eta_s"])
        fraction = 100.0 * state["completed"] / self.total if self.total else 100.0
        self.printer(
            f"{label} 场景={state['completed']}/{self.total} ({fraction:.1f}%) "
            f"本次新增={self.new_completed} 复用={self.cached} "
            f"已用={duration_text(state['elapsed_s'], round_up=False)} 预计剩余={eta}", flush=True,
        )
        self.last_reported_at = now
        self.last_reported_new = self.new_completed
        return True
