"""声纹注册 FastAPI 入口。"""
import argparse
import logging
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from modules.utils import setup_logging
from voiceprint_server.api.enroll import router as enroll_router


def create_app():
    app = FastAPI(title="StreamScore Voiceprint Enrollment", version="1.0")
    app.include_router(enroll_router)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="StreamScore Voiceprint Enrollment Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    import uvicorn
    logger.info("Starting Voiceprint Enrollment Server on %s:%d", args.host, args.port)
    uvicorn.run(
        "voiceprint_server.server:create_app",
        host=args.host,
        port=args.port,
        factory=True,
    )


app = create_app()

if __name__ == "__main__":
    main()
