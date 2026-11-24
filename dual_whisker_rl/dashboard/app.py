"""只监听本机的管理 API 和静态页面。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import threading
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .service import Service


class CreateTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=8, max_length=100)
    name: str = Field(default="", max_length=80)
    variant: Literal["original", "improved"] = "improved"
    target_steps: int = Field(strict=True, gt=0, le=10**12)
    seed: int | None = Field(default=None, strict=True, ge=0, le=2**32 - 1)
    source_task_id: str | None = None
    checkpoint_id: str | None = None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=8, max_length=100)
    action: Literal["set_target", "pause", "resume", "stop", "takeover"]
    target_steps: int | None = Field(default=None, strict=True, gt=0, le=10**12)
    checkpoint_id: str | None = None


def create_app(service: Service, *, background=True):
    stop = threading.Event()
    token = secrets.token_urlsafe(32)

    def poll():
        while not stop.is_set():
            try:
                service.tick()
            except Exception as exc:
                service.last_error = str(exc)
            stop.wait(5)

    @asynccontextmanager
    async def lifespan(app):
        thread = None
        if background:
            thread = threading.Thread(target=poll, name="training-scheduler", daemon=True)
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=5)

    app = FastAPI(title="双触须训练管理", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"error": "仅允许本机访问"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": "拒绝跨站管理操作"}, status_code=403)
            if request.cookies.get("dashboard_session") != token or request.headers.get("x-csrf-token") != token:
                return JSONResponse({"error": "页面会话已失效，请刷新页面"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        response = JSONResponse({"csrf_token": token, "max_concurrent": 2})
        response.set_cookie("dashboard_session", token, httponly=True, samesite="strict")
        return response

    @app.get("/api/overview")
    def overview():
        return service.overview()

    @app.get("/api/tasks")
    def tasks():
        return service.tasks()

    @app.post("/api/tasks")
    def create(body: CreateTask):
        return service.create(body.model_dump(exclude_none=True))

    @app.get("/api/tasks/{task_id}")
    def detail(task_id: str):
        return service.detail(task_id)

    @app.get("/api/tasks/{task_id}/metrics")
    def metrics(task_id: str, tags: str = ""):
        return service.metrics(task_id, [s for s in tags.split(",") if s][:8] or None)

    @app.get("/api/tasks/{task_id}/logs")
    def logs(task_id: str, attempt_id: str | None = None, offset: int | None = None):
        return service.logs(task_id, attempt_id, offset)

    @app.get("/api/tasks/{task_id}/checkpoints")
    def checkpoints(task_id: str):
        return service.checkpoints(task_id)

    @app.get("/api/tasks/{task_id}/takeover-preview")
    def preview(task_id: str):
        return service.takeover_preview(task_id)

    @app.post("/api/tasks/{task_id}/actions")
    def action(task_id: str, body: Action):
        return service.action(task_id, body.model_dump(exclude_none=True))

    return app
