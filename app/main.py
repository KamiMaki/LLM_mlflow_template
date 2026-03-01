"""應用程式入口 — FastAPI 工廠 + uvicorn 啟動。

Usage:
    # 直接執行
    python -m app.main

    # 使用 Hydra override
    python -m app.main env=prod

    # 使用 uvicorn (支援 hot reload)
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """應用程式生命週期：啟動時初始化 config、logger、MLflow。"""
    from app.utils.config import init_config
    from app.logger import setup_logging
    from app.tracking.setup import init_mlflow

    cfg = init_config()
    setup_logging(cfg)
    init_mlflow(cfg)

    logger.info("Application started")
    yield
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    """建立 FastAPI 應用程式。"""
    application = FastAPI(
        title="LLM Service",
        description="LLM 後端服務模板",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 註冊基礎 router
    from app.api.router import base_router
    application.include_router(base_router)

    return application


# Module-level instance for `uvicorn app.main:app`
app = create_app()


def start():
    """啟動 uvicorn 服務。"""
    from app.utils.config import init_config

    overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    cfg = init_config(overrides=overrides)

    host = cfg.api.host if hasattr(cfg, "api") else "0.0.0.0"
    port = cfg.api.port if hasattr(cfg, "api") else 8000
    workers = cfg.api.workers if hasattr(cfg, "api") else 1

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    start()
