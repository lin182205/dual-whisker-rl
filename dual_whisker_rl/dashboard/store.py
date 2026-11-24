"""SQLite 注册表；训练进程的生命周期独立于网页服务。"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time


class Store:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(Path(directory) / "dashboard.sqlite"), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, import_key TEXT UNIQUE, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, task_id TEXT, at REAL, action TEXT, detail TEXT);
        """)
        self.db.commit()

    def tasks(self):
        with self.lock:
            return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM tasks ORDER BY rowid")]

    def get(self, task_id):
        with self.lock:
            row = self.db.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError("任务不存在")
            return json.loads(row[0])

    def save(self, task):
        with self.lock, self.db:
            self.db.execute("INSERT INTO tasks VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                            (task["id"], task.get("import_key"), json.dumps(task, ensure_ascii=False)))

    def event(self, task_id, action, detail=None):
        with self.lock, self.db:
            self.db.execute("INSERT INTO events(task_id,at,action,detail) VALUES (?,?,?,?)",
                            (task_id, time.time(), action, json.dumps(detail, ensure_ascii=False)))

    def events(self, task_id):
        with self.lock:
            rows = self.db.execute("SELECT at,action,detail FROM events WHERE task_id=? ORDER BY id DESC LIMIT 50", (task_id,))
            return [{"at": a, "action": b, "detail": json.loads(c)} for a, b, c in rows]

    def previous(self, request_id, fingerprint):
        with self.lock:
            row = self.db.execute("SELECT fingerprint,result FROM requests WHERE id=?", (request_id,)).fetchone()
            if row and row[0] != fingerprint:
                raise ValueError("请求标识已用于其他操作")
            return json.loads(row[1]) if row else None

    def remember(self, request_id, fingerprint, result):
        with self.lock, self.db:
            self.db.execute("INSERT INTO requests VALUES (?,?,?)", (request_id, fingerprint, json.dumps(result)))

    def close(self):
        with self.lock:
            self.db.close()
