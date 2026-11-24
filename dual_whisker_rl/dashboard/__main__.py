"""启动面板；单服务进程确保队列不会重复派发。"""
import argparse
from pathlib import Path
import sys

from .app import create_app
from .service import Service


def main():
    parser = argparse.ArgumentParser(description="双触须训练管理面板，仅本机监听")
    parser.add_argument("--directory", type=Path, default=Path("results/training_dashboard"))
    parser.add_argument("--training-python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--import-manifest", action="append", default=[])
    parser.add_argument("--allow-cpu", action="store_true", help="仅本地隔离验证时允许CPU训练")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    lock = (args.directory / "service.lock").open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("此管理目录已有面板服务运行")
    service = Service(args.directory, training_python=args.training_python, require_gpu=not args.allow_cpu)
    for manifest in args.import_manifest:
        service.import_manifest(manifest)
    import uvicorn
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, workers=1, access_log=False)
    service.store.close()
    lock.close()


if __name__ == "__main__":
    main()
