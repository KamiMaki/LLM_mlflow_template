"""基礎 Router — health check 與 readiness probe。"""

from __future__ import annotations

from fastapi import APIRouter

base_router = APIRouter(tags=["system"])


@base_router.get("/health")
async def health():
    """Health check endpoint。"""
    return {"status": "ok"}


@base_router.get("/ready")
async def ready():
    """Readiness probe — 確認服務已初始化。"""
    try:
        from app.utils.config import get_config
        get_config()
        return {"status": "ready"}
    except RuntimeError:
        return {"status": "not_ready", "reason": "Config not initialized"}
