"""单实例任务管理、旧任务导入与 FIFO 调度。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import psutil
import yaml

from .collect import LogProgress, Metrics, Resources, log_page
from .files import alive, atomic_json, checkpoint_data, identity, now, read_json, rooted
from .store import Store


ACTIVE = {"starting", "running", "pausing", "stopping", "unresponsive"}
TERMINAL = {"completed", "stopped", "failed"}
PRESETS = {"original": "mobile_rooms_hybrid_history.yaml", "improved": "mobile_rooms_hybrid_history_improved.yaml"}


class Service:
    def __init__(self, directory, project_root=None, training_python=None, *, require_gpu=True):
        self.root = Path(directory).resolve()
        self.project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.python = str(training_python or sys.executable)
        self.require_gpu = require_gpu
        self.store = Store(self.root)
        self.lock = threading.RLock()
        self.metrics_lock = threading.Lock()
        self.progress = LogProgress()
        self.metrics_reader = Metrics()
        self.resources = Resources(self.root)
        self.children = {}
        self.cp_cache = {}
        self.last_error = None

    def import_manifest(self, filename):
        manifest = read_json(filename)
        if not manifest or not isinstance(manifest.get("runs"), list):
            raise ValueError("启动清单无效")
        with self.lock:
            existing = {t.get("import_key") for t in self.store.tasks()}
            for run in manifest["runs"]:
                key = str(Path(filename).resolve()) + "#" + run["session"]
                if key in existing:
                    continue
                cwd = Path(run["cwd"]).resolve()
                meta_path = run.get("metadata_path")
                if not meta_path:
                    raise ValueError("清单缺少 metadata_path，请使用已验证的启动清单")
                meta = read_json(meta_path)
                if not meta:
                    raise ValueError("无法读取原训练 metadata")
                who = None
                for p in psutil.process_iter(["pid", "cmdline", "cwd"]):
                    try:
                        if p.info["cmdline"] == run["command"] and Path(p.info["cwd"]).resolve() == cwd:
                            who = identity(p.pid)
                            break
                    except (psutil.Error, TypeError, OSError):
                        continue
                variant = "improved" if meta["config"].get("blank_loop_penalty", 0) else "original"
                task_id = uuid.uuid5(uuid.NAMESPACE_URL, key).hex
                attempt = {"id": "imported", "legacy": True, "cwd": str(cwd),
                           "metadata": str(meta_path), "log": run["train_log"], "identity": who,
                           "starting_steps": run["starting_timesteps"], "label": "接入前训练",
                           "checkpoint_dir": str(rooted(cwd, meta["checkpoint_dir"])),
                           "tensorboard_dir": str(rooted(cwd, meta["tensorboard_run_dir"])),
                           "model_path": str(rooted(cwd, meta["model_path"])),
                           "source_commit": run.get("source_commit")}
                label = "原版" if variant == "original" else "改进版"
                description = f"{run['starting_timesteps'] / 10000:g}万步续训" if run["starting_timesteps"] else "从零训练"
                task = {"id": task_id, "import_key": key, "name": label + " · " + description,
                        "variant": variant, "mode": "legacy", "state": "running" if alive(who) else "unknown",
                        "created": now(), "start_steps": int(run["starting_timesteps"]),
                        "current_steps": int(run["starting_timesteps"]), "target_steps": int(run["target_timesteps"]),
                        "seed": meta["seed"], "n_envs": meta["n_envs"], "n_steps": meta["ppo"]["n_steps"],
                        "batch_size": meta["ppo"]["batch_size"], "config": meta["config"], "source_meta": meta,
                        "attempts": [attempt], "legacy_session": run["session"], "revision": 0,
                        "resume_checkpoint": None, "queued_at": None}
                self.store.save(task)
                self.store.event(task_id, "imported", {"manifest": str(filename)})
            self._refresh_all()

    def _control_path(self, task):
        return self.root / "tasks" / task["id"] / "control.json"

    def _write_control(self, task, desired):
        task["revision"] = task.get("revision", 0) + 1
        task["desired_state"] = desired
        self.store.save(task)
        atomic_json(self._control_path(task), {"revision": task["revision"], "desired_state": desired,
                                             "target_steps": task["target_steps"], "requested_at": now()})

    def _reconcile_control(self, task):
        control = read_json(self._control_path(task), {})
        if control.get("revision") != task.get("revision"):
            atomic_json(self._control_path(task), {"revision": task["revision"],
                        "desired_state": task.get("desired_state", "running"),
                        "target_steps": task["target_steps"], "requested_at": now()})

    def _refresh_all(self):
        for task in self.store.tasks():
            self._refresh(task)
            self.store.save(task)

    def _refresh(self, task):
        if not task["attempts"]:
            return
        attempt = task["attempts"][-1]
        sample = self.progress.read(attempt["log"])
        if sample.get("current_steps") is not None:
            task["current_steps"] = sample["current_steps"]
        for key in ("cycle_seconds", "cycle_samples", "last_update"):
            if key in sample:
                task[key] = sample[key]
        if task["mode"] == "legacy":
            if sample.get("logged_target"):
                task["target_steps"] = sample["logged_target"]
            if alive(attempt.get("identity")):
                task["state"] = "running"
            else:
                meta = read_json(attempt["metadata"], {})
                if meta.get("training_complete") and (meta.get("final_num_timesteps") or 0) >= task["target_steps"]:
                    task.update(state="completed", current_steps=meta["final_num_timesteps"])
                elif sample.get("traceback"):
                    task["state"] = "failed"
                else:
                    task["state"] = "failed" if attempt.get("identity") else "unknown"
            return
        if attempt.get("legacy"):
            # 接管只创建新的运行段，旧段没有 managed status.json。
            return
        state = read_json(Path(attempt["directory"]) / "status.json", {})
        if not attempt.get("identity") and state.get("identity"):
            attempt["identity"] = state["identity"]
        if state.get("current_steps") is not None:
            task["current_steps"] = state["current_steps"]
            task["last_update"] = state.get("at", task.get("last_update"))
        live = alive(attempt.get("identity"))
        child = self.children.get(task["id"])
        if child is not None and child.poll() is not None:
            self.children.pop(task["id"], None)
            live = False
        if live:
            if task["state"] not in {"pausing", "stopping"}:
                task["state"] = "running" if state.get("state") == "running" else "starting"
        elif task["state"] in ACTIVE:
            if not attempt.get("identity") and now() - attempt.get("started", 0) < 45:
                return
            outcome = state.get("state")
            if outcome in {"paused", "stopped", "completed"}:
                prior_state = task["state"]
                task["state"] = outcome
                task["resume_checkpoint"] = state.get("checkpoint")
                attempt["end_steps"] = task["current_steps"]
                # 处理结束落盘与目标修改并发的窄窗口，不能把新目标提前标为完成。
                if outcome == "completed" and task["current_steps"] < task["target_steps"]:
                    task["state"] = {"pausing": "paused", "stopping": "stopped"}.get(prior_state, "queued")
                    if task["state"] == "queued":
                        task["queued_at"] = now()
                        self._write_control(task, "running")
            else:
                task["state"] = "failed"
                task["error"] = state.get("error", "训练进程已退出，请查看日志和最近检查点")
        if state.get("checkpoint"):
            attempt["last_checkpoint"] = state["checkpoint"]

    def tasks(self):
        with self.lock:
            tasks = self.store.tasks()
            for task in tasks:
                size = task["n_envs"] * task["n_steps"]
                task["rollout_size"] = size
                task["session_progress"] = max(0, min(1, (task["current_steps"] - task["start_steps"]) / max(1, task["target_steps"] - task["start_steps"])))
                task["total_progress"] = min(1, task["current_steps"] / max(1, task["target_steps"]))
                active = task["state"] in ACTIVE
                average = task.get("cycle_seconds")
                task["eta_seconds"] = math.ceil(max(0, task["target_steps"] - task["current_steps"]) / size) * average if active and average and task.get("cycle_samples", 0) >= 3 else None
                task["steps_per_second"] = size / average if active and average else None
                task["stale"] = active and now() - task.get("last_update", task["created"]) > max(180, 3 * (average or 60))
                # 配置、命令与进程身份只放在详情，列表保持轻量。
                for key in ("source_meta", "config", "attempts"):
                    task.pop(key, None)
            return tasks

    def overview(self):
        tasks = self.tasks()
        return {"tasks": tasks, "resources": self.resources.read(), "max_concurrent": 2,
                "last_error": self.last_error, "at": now()}

    def detail(self, task_id):
        with self.lock:
            task = self.store.get(task_id)
            task["events"] = self.store.events(task_id)
            return task

    def checkpoints(self, task_id):
        with self.lock:
            task = self.store.get(task_id)
            result = []
            for attempt in task["attempts"]:
                path = Path(attempt["checkpoint_dir"])
                for file in path.glob("*_steps.zip"):
                    try:
                        stat = file.stat()
                        cache_key = (str(file), stat.st_size, stat.st_mtime_ns)
                        if cache_key not in self.cp_cache:
                            data = checkpoint_data(file)
                            self.cp_cache[cache_key] = {"steps": data["num_timesteps"]}
                        entry_id = hashlib.sha256(str(file).encode()).hexdigest()[:24]
                        result.append({"id": entry_id, "name": file.name, "steps": self.cp_cache[cache_key]["steps"],
                                       "size": stat.st_size, "at": stat.st_mtime, "path": str(file)})
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                        continue
            if len(self.cp_cache) > 2000:
                self.cp_cache.clear()
            return sorted(result, key=lambda row: (row["steps"], row["at"]), reverse=True)

    def _checkpoint(self, task_id, checkpoint_id=None):
        entries = self.checkpoints(task_id)
        for item in entries:
            if checkpoint_id is None or item["id"] == checkpoint_id:
                return item
        raise ValueError("尚无有效检查点，暂不能续训或接管")

    def takeover_preview(self, task_id):
        task = self.store.get(task_id)
        if task["mode"] != "legacy":
            raise ValueError("此任务已由面板管理")
        cp = self._checkpoint(task_id)
        return {"checkpoint": cp, "current_steps": task["current_steps"],
                "rollback_steps": max(0, task["current_steps"] - cp["steps"])}

    def logs(self, task_id, attempt_id=None, offset=None):
        task = self.store.get(task_id)
        attempts = task["attempts"]
        if not attempts:
            return {"text": "任务尚未启动", "offset": 0, "truncated": False}
        attempt = next((a for a in attempts if a["id"] == attempt_id), None) if attempt_id else attempts[-1]
        if not attempt:
            raise ValueError("运行段不存在")
        return log_page(attempt["log"], offset)

    def metrics(self, task_id, tags=None):
        task = self.store.get(task_id)
        with self.metrics_lock:
            return self.metrics_reader.read(task, tags)

    def create(self, body):
        with self.lock:
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
                raise ValueError("创建任务缺少有效请求标识")
            fingerprint = json.dumps(body, sort_keys=True)
            prior = self.store.previous(request_id, fingerprint)
            if prior is not None:
                return prior
            task_id = uuid.uuid5(uuid.NAMESPACE_URL, "training-task:" + request_id).hex
            existing = next((t for t in self.store.tasks() if t["id"] == task_id), None)
            if existing:
                if existing.get("creation_request") != fingerprint:
                    raise ValueError("请求标识已用于其他创建操作")
                return {"task_id": task_id, "state": existing["state"], "target_steps": existing["target_steps"]}
            variant = body.get("variant", "improved")
            if variant not in PRESETS:
                raise ValueError("请选择原版或改进版")
            target = self._integer(body.get("target_steps"), "累计目标")
            source = body.get("source_task_id")
            cp, data, meta = None, {}, {}
            config = yaml.safe_load((self.project / "configs" / PRESETS[variant]).read_text(encoding="utf-8"))
            if source:
                parent = self.store.get(source)
                if parent["variant"] != variant:
                    raise ValueError("续训必须沿用源检查点的环境版本")
                cp = self._checkpoint(source, body.get("checkpoint_id"))
                data = checkpoint_data(cp["path"], crc=True)
                config, meta = parent["config"], parent.get("source_meta", {})
                if parent.get("attempts"):
                    meta = read_json(parent["attempts"][-1].get("metadata", ""), meta)
                meta = {**meta, "seed": meta.get("seed", parent["seed"])}
            start = data.get("num_timesteps", 0)
            n_envs, n_steps, batch = (data.get("n_envs", 4), data.get("n_steps", 1250), data.get("batch_size", 250))
            if target <= start:
                raise ValueError("累计目标必须大于检查点步数")
            size = n_envs * n_steps
            target = start + math.ceil((target - start) / size) * size
            seed = body.get("seed")
            seed = (meta.get("seed", secrets.randbelow(2**31)) if source else secrets.randbelow(2**31)) if seed is None else self._integer(seed, "随机种子", zero=True)
            if seed > 2**32 - 1:
                raise ValueError("随机种子不能超过4294967295")
            task = {"id": task_id, "creation_request": fingerprint, "name": str(body.get("name") or ("原版训练" if variant == "original" else "改进版训练"))[:80],
                    "variant": variant, "mode": "managed", "state": "queued", "created": now(), "queued_at": now(),
                    "start_steps": start, "current_steps": start, "target_steps": target, "seed": seed,
                    "n_envs": n_envs, "n_steps": n_steps, "batch_size": batch, "config": config,
                    "source_meta": meta, "resume_checkpoint": cp["path"] if cp else None,
                    "parent_task": source, "attempts": [], "revision": 0}
            self.store.save(task)
            self._write_control(task, "running")
            self.store.save(task)
            self.store.event(task["id"], "created", {"target_steps": target, "parent_task": source})
            result = {"task_id": task["id"], "state": "queued", "target_steps": target}
            self.store.remember(request_id, fingerprint, result)
            return result

    @staticmethod
    def _integer(value, label, zero=False):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
            raise ValueError(label + "必须为" + ("非负" if zero else "正") + "整数")
        return value

    def action(self, task_id, body):
        with self.lock:
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
                raise ValueError("操作缺少有效请求标识")
            fingerprint = json.dumps([task_id, body], sort_keys=True)
            previous = self.store.previous(request_id, fingerprint)
            if previous is not None:
                return previous
            task = self.store.get(task_id)
            self._refresh(task)
            action = body.get("action")
            if task["mode"] == "legacy":
                if action != "takeover":
                    raise ValueError("此任务目前仅监控，请先手动接管控制")
                result = self._takeover(task, body.get("checkpoint_id"))
            elif action == "set_target":
                if task["state"] in TERMINAL or task["state"] == "stopping":
                    raise ValueError("已结束的任务请新建续训任务")
                target = self._integer(body.get("target_steps"), "累计目标")
                if target <= task["current_steps"]:
                    raise ValueError("目标不能低于当前进度，请使用停止操作")
                size = task["n_envs"] * task["n_steps"]
                task["target_steps"] = task["start_steps"] + math.ceil((target - task["start_steps"]) / size) * size
                desired = "paused" if task["state"] in {"paused", "pausing"} else "running"
                self._write_control(task, desired)
                result = {"target_steps": task["target_steps"]}
            elif action == "pause":
                if task["state"] not in {"running", "starting", "queued"}:
                    raise ValueError("当前状态不能暂停")
                task["state"] = "paused" if task["state"] == "queued" else "pausing"
                self._write_control(task, "paused")
                result = {"state": task["state"]}
            elif action == "resume":
                if task["state"] != "paused":
                    raise ValueError("只有暂停任务可以继续；其他任务请新建续训")
                task.update(state="queued", queued_at=now())
                self._write_control(task, "running")
                result = {"state": "queued"}
            elif action == "stop":
                if task["state"] not in ACTIVE | {"queued", "paused"}:
                    raise ValueError("此任务已结束")
                task["state"] = "stopping" if task["state"] in ACTIVE else "stopped"
                self._write_control(task, "stopped")
                result = {"state": task["state"]}
            else:
                raise ValueError("未知操作")
            self.store.save(task)
            self.store.event(task_id, action, result)
            self.store.remember(request_id, fingerprint, result)
            return result

    def _stage(self, task):
        attempt_id = f"{len(task['attempts']) + 1:03d}_{uuid.uuid4().hex[:8]}"
        directory = self.root / "tasks" / task["id"] / "attempts" / attempt_id
        directory.mkdir(parents=True, exist_ok=False)
        spec = {"task": task, "control_path": str(self._control_path(task)), "require_gpu": self.require_gpu}
        source = task.get("resume_checkpoint")
        if source:
            frozen = directory / "source" / Path(source).name
            frozen.parent.mkdir()
            shutil.copy2(source, frozen)
            checkpoint_data(frozen, crc=True)
            spec["resume_checkpoint"] = str(frozen)
            spec["source_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
        atomic_json(directory / "spec.json", spec)
        (directory / "config.yaml").write_text(yaml.safe_dump(task["config"], allow_unicode=True, sort_keys=False), encoding="utf-8")
        return directory

    def _worker_command(self, directory, validate=False):
        return [self.python, "-u", "-m", "dual_whisker_rl.dashboard.worker", "--attempt-dir", str(directory)] + (["--validate-only"] if validate else [])

    def _worker_env(self):
        return {**os.environ, "PYTHONPATH": str(self.project), "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

    def _start(self, task, directory=None):
        directory = directory or self._stage(task)
        attempt = {"id": directory.name, "directory": str(directory), "cwd": str(self.project),
                   "log": str(directory / "train.log"), "metadata": str(directory / "logs/run_metadata.json"),
                   "checkpoint_dir": str(directory / "checkpoints"), "tensorboard_dir": str(directory / "tensorboard/run"),
                   "model_path": str(directory / "final_model.zip"), "starting_steps": task["current_steps"],
                   "identity": None, "started": now(), "label": "受管理训练"}
        # 先登记启动槽位，进程再发布自己的身份，服务重启时不会重复启动。
        task["attempts"].append(attempt)
        task.update(state="starting", last_update=now())
        self.store.save(task)
        try:
            with (directory / "train.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(self._worker_command(directory), cwd=self.project, env=self._worker_env(),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True)
            who = None
            for _ in range(20):
                who = identity(child.pid)
                if who or child.poll() is not None:
                    break
                time.sleep(.02)
            if not who:
                raise RuntimeError("训练进程在启动时退出")
            attempt["identity"] = who
            self.children[task["id"]] = child
            self.store.save(task)
            atomic_json(directory / "launch.json", {"command": self._worker_command(directory), "identity": who,
                        "source_commit": self._commit(), "at": now()})
            self.store.event(task["id"], "started", {"attempt_id": directory.name, "pid": child.pid})
        except Exception:
            task.update(state="failed", error="任务启动失败，请检查训练环境")
            self.store.save(task)
            raise

    def _commit(self):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.project, text=True, timeout=3).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _takeover(self, task, checkpoint_id):
        if os.name != "posix":
            raise ValueError("旧任务接管只在训练服务器提供")
        cp = self._checkpoint(task["id"], checkpoint_id)
        old = task["attempts"][-1]
        who = old.get("identity")
        current = identity(who["pid"]) if alive(who) else None
        if not current or current["command"] != who["command"] or current["cwd"] != who["cwd"]:
            raise ValueError("原进程身份已变化，已取消接管")
        pgid = os.getpgid(who["pid"])
        if pgid == os.getpgrp() or pgid <= 1:
            raise ValueError("无法安全识别训练进程组")
        # 模型及环境先验收，失败时原训练继续；不执行网页提供的路径或命令。
        task["resume_checkpoint"] = cp["path"]
        staged = self._stage(task)
        result = subprocess.run(self._worker_command(staged, validate=True), cwd=self.project, env=self._worker_env(),
                                capture_output=True, text=True, timeout=90)
        (staged / "validation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise ValueError("检查点接管预检失败，原训练未停止；请查看管理目录中的 validation.log")
        fresh = identity(who["pid"]) if alive(who) else None
        if not fresh or fresh["command"] != who["command"] or os.getpgid(who["pid"]) != pgid:
            raise ValueError("预检期间原进程已变化，接管取消")
        family = [psutil.Process(who["pid"]), *psutil.Process(who["pid"]).children(recursive=True)]
        os.killpg(pgid, signal.SIGTERM)
        _, survivors = psutil.wait_procs(family, timeout=10)
        if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in survivors):
            raise ValueError("原进程尚未全部退出，未启动新的训练，请检查原任务")
        old["end_steps"] = cp["steps"]
        old["rollback_from_steps"] = task["current_steps"]
        task.update(mode="managed", state="queued", current_steps=cp["steps"], queued_at=now())
        task["resume_checkpoint"] = str(staged / "source" / Path(cp["path"]).name)
        self._write_control(task, "running")
        self.store.save(task)
        return {"state": "queued", "checkpoint_steps": cp["steps"], "rollback_steps": max(0, old["rollback_from_steps"] - cp["steps"])}

    def tick(self):
        with self.lock:
            for task in self.store.tasks():
                if task["mode"] == "managed":
                    self._reconcile_control(task)
            self._refresh_all()
            tasks = self.store.tasks()
            active = sum(t["state"] in ACTIVE or (t["mode"] == "legacy" and t["state"] == "unknown") for t in tasks)
            queue = sorted((t for t in tasks if t["state"] == "queued"), key=lambda t: t["queued_at"])
            for task in queue[:max(0, 2 - active)]:
                try:
                    self._start(task)
                except Exception as exc:
                    self.last_error = str(exc)
                    task.update(state="failed", error=str(exc))
                    self.store.save(task)
                    self.store.event(task["id"], "failed", {"error": str(exc)})
