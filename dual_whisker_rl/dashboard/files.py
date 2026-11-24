"""管理文件、检查点摘要与进程身份校验。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid
import zipfile

import psutil


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def rooted(cwd, path):
    value = Path(path)
    return value if value.is_absolute() else Path(cwd) / value


def checkpoint_data(path, *, crc=False):
    """只读 JSON 摘要，不在 Web 服务中反序列化模型或执行 pickle。"""
    with zipfile.ZipFile(path) as archive:
        if crc and archive.testzip() is not None:
            raise ValueError("检查点校验失败")
        data = json.loads(archive.read("data"))
    step = data.get("num_timesteps")
    if not isinstance(step, int) or step < 0:
        raise ValueError("检查点缺少有效累计步数")
    return data


def proc_token(pid):
    """Linux原始启动时钟计数不使用会随校时变化的btime墙钟基准。"""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": stat.rsplit(")", 1)[1].split()[19]}
    except (OSError, ValueError, IndexError):
        return None


def identity(pid):
    try:
        before = proc_token(pid)
        p = psutil.Process(int(pid))
        result = {"pid": p.pid, "created": p.create_time(), "cwd": p.cwd(),
                  "command": p.cmdline()}
        after = proc_token(pid)
        if before != after:
            return None
        if after:
            result["proc_token"] = after
        return result
    except (psutil.Error, OSError, ValueError, TypeError):
        return None


def alive(who):
    if not who:
        return False
    try:
        p = psutil.Process(int(who["pid"]))
        if who.get("proc_token"):
            return proc_token(who["pid"]) == who["proc_token"] and p.status() != psutil.STATUS_ZOMBIE
        return abs(p.create_time() - who["created"]) < .01 and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError, TypeError):
        return False


def now():
    return time.time()
