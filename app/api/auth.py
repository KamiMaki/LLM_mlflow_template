"""Bearer Token 認證 — 簡易 API 金鑰驗證。

Usage:
    from app.api.auth import require_auth

    @router.get("/protected")
    async def protected(user=Depends(require_auth)):
        return {"message": "authenticated"}
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.logger import get_logger

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_auth_token() -> str:
    """從設定中取得 API auth token。"""
    try:
        from app.utils.config import get_config

        cfg = get_config()
        return cfg.api.auth_token if hasattr(cfg.api, "auth_token") else ""
    except RuntimeError:
        return ""


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """驗證 Bearer token。

    Returns:
        通過驗證的 token 字串。

    Raises:
        HTTPException 401: 缺少或無效的 token。
    """
    expected = _get_auth_token()

    # 若未設定 auth_token，跳過驗證
    if not expected:
        return ""

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != expected:
        logger.warning("Invalid auth token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
