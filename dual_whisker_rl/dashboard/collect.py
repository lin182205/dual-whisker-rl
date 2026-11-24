"""增量读取训练输出和 TensorBoard；只读取，不加载模型。"""
from __future__ import annotations

from collections import OrderedDict, deque
import math
from pathlib import Path
import re
import subprocess
import time

import psutil

from .files import rooted


CYCLE = re.compile(r"轮结束.*?本轮总耗时=([\d.]+)秒.*?训练步数=(\d+)/(\d+)")
DEFAULT_TAGS = ["rollout/success_rate_last_500", "rollout/ep_rew_mean",
                "rollout/blank_loop_ratio", "rollout/collision_rate",
                "diagnostics/mean_blank_age_steps", "train/approx_kl"]


class LogProgress:
    def __init__(self):
        self.files = {}

    def read(self, filename):
        path = Path(filename)
        if not path.is_file():
            return {}
        stat = path.stat()
        key = str(path)
        item = self.files.setdefault(key, {"offset": max(0, stat.st_size - 262144),
                                           "pending": b"", "durations": deque(maxlen=5)})
        if stat.st_size < item["offset"] or item.get("inode", stat.st_ino) != stat.st_ino:
            item.update(offset=0, pending=b"", durations=deque(maxlen=5))
            item.pop("current_steps", None)
        item["inode"] = stat.st_ino
        with path.open("rb") as stream:
            stream.seek(item["offset"])
            block = stream.read(524288)
            item["offset"] = stream.tell()
        lines = (item["pending"] + block).split(b"\n")
        item["pending"] = lines.pop()[-16384:]
        for line in lines:
            text = line.decode("utf-8", errors="replace")
            found = CYCLE.search(text)
            if found:
                duration, step, target = found.groups()
                item.update(current_steps=int(step), logged_target=int(target), last_update=stat.st_mtime)
                item["durations"].append(float(duration))
            if "Traceback (most recent call last)" in text:
                item["traceback"] = True
        result = {k: v for k, v in item.items() if k not in {"offset", "pending", "durations", "inode"}}
        if item["durations"]:
            result["cycle_seconds"] = sum(item["durations"]) / len(item["durations"])
            result["cycle_samples"] = len(item["durations"])
        return result


def log_page(filename, offset=None, limit=32000):
    path = Path(filename)
    if not path.is_file():
        return {"text": "日志尚未生成", "offset": 0, "truncated": False}
    size = path.stat().st_size
    start = max(0, size - limit) if offset is None else max(0, min(int(offset), size))
    with path.open("rb") as stream:
        stream.seek(start)
        text = stream.read(limit).decode("utf-8", errors="replace")
        end = stream.tell()
    return {"text": text, "offset": end, "truncated": end < size}


def thin(points, limit=1000):
    """保留首尾与分桶极值，避免长曲线吞没短时异常。"""
    if len(points) <= limit:
        return points
    width = max(1, math.ceil((len(points) - 2) / ((limit - 2) // 2)))
    picked = [points[0]]
    for start in range(1, len(points) - 1, width):
        chunk = points[start:min(start + width, len(points) - 1)]
        extremes = {min(range(len(chunk)), key=lambda i: chunk[i][1]),
                    max(range(len(chunk)), key=lambda i: chunk[i][1])}
        picked.extend(chunk[i] for i in sorted(extremes))
    return (picked + [points[-1]])[:limit - 1] + [points[-1]] if len(picked) + 1 > limit else picked + [points[-1]]


class Metrics:
    def __init__(self):
        self.cache = OrderedDict()

    def directory(self, path):
        key = str(path)
        if not Path(path).is_dir():
            return None
        if key not in self.cache:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            self.cache[key] = [EventAccumulator(key, size_guidance={"scalars": 2000,
                              "histograms": 1, "images": 1, "audio": 1, "tensors": 1}), 0.0]
        entry = self.cache.pop(key)
        self.cache[key] = entry
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        if time.monotonic() - entry[1] >= 15:
            entry[0].Reload()
            entry[1] = time.monotonic()
        return entry[0]

    def read(self, task, tags=None):
        tags = (tags or DEFAULT_TAGS)[:8]
        available, series, boundaries = set(), {tag: {} for tag in tags}, []
        for attempt in task.get("attempts", []):
            tb = attempt.get("tensorboard_dir")
            if not tb:
                continue
            try:
                accumulator = self.directory(tb)
                if accumulator is None:
                    continue
                names = accumulator.Tags().get("scalars", [])
                available.update(names)
                cutoff = attempt.get("end_steps")
                for tag in tags:
                    if tag not in names:
                        continue
                    for point in accumulator.Scalars(tag):
                        if math.isfinite(point.value) and (cutoff is None or point.step <= cutoff):
                            series[tag][point.step] = float(point.value)
                if attempt.get("starting_steps") is not None:
                    boundaries.append({"step": attempt["starting_steps"], "label": attempt.get("label", "运行段")})
            except (OSError, ValueError, KeyError):
                # 正在写入的最后一条事件可能不完整，保留已采集数据，下次重试。
                continue
        return {"series": {tag: thin(sorted(values.items())) for tag, values in series.items()},
                "available_tags": sorted(available), "boundaries": boundaries}


class Resources:
    def __init__(self, disk):
        self.disk = str(disk)
        self.previous = None
        self.cached = {}
        self.at = 0.0

    def read(self):
        stamp = time.monotonic()
        if self.cached and stamp - self.at < 5:
            return self.cached
        cpu_count = psutil.cpu_count() or 1
        memory = psutil.virtual_memory()
        total, used = memory.total, memory.used
        cpu_percent = psutil.cpu_percent()
        cgroup = Path("/sys/fs/cgroup")
        try:
            quota, period = (cgroup / "cpu.max").read_text().split()
            if quota != "max":
                cpu_count = float(quota) / float(period)
            stats = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())
            usage = int(stats["usage_usec"]) / 1e6
            if self.previous:
                cpu_percent = max(0, min(100, (usage - self.previous[1]) / (stamp - self.previous[0]) / cpu_count * 100))
            else:
                cpu_percent = None
            self.previous = (stamp, usage)
            maximum = (cgroup / "memory.max").read_text().strip()
            if maximum != "max":
                total = int(maximum)
                used = int((cgroup / "memory.current").read_text())
        except (OSError, ValueError, KeyError):
            pass
        disk = psutil.disk_usage(self.disk)
        gpu = None
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                name, load, allocated, capacity = result.stdout.splitlines()[0].rsplit(",", 3)
                gpu = {"name": name.strip(), "percent": float(load), "used_mb": float(allocated), "total_mb": float(capacity)}
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
        self.cached = {"cpu_cores": cpu_count, "cpu_percent": round(cpu_percent, 1) if cpu_percent is not None else None,
                       "memory_used": used, "memory_total": total, "disk_used": disk.used,
                       "disk_total": disk.total, "gpu": gpu, "at": time.time()}
        self.at = stamp
        return self.cached
