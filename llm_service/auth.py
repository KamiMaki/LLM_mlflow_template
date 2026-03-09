"""Token Exchange — 使用初始 token (J1) 換取 API 存取 token (J2)。

支援自動快取與過期重新取得，適用於內部 API 需要先驗證換 token 的場景。

Usage:
    from llm_service.auth import TokenExchanger

    exchanger = TokenExchanger(
        auth_url="https://auth.internal.com/token",
        auth_token="your-j1-token",
    )
    j2_token = exchanger.get_token()  # 自動快取，過期自動重取
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from loguru import logger


class TokenExchanger:
    """Token 交換器：拿 J1 token 換取 J2 token，自動快取與刷新。

    流程：POST auth_url，帶 J1 token（Authorization header），
    回應中取得 J2 token 和可選的 expires_in。

    Attributes:
        auth_url: 驗證端點 URL。
        auth_token: 初始 token (J1)。
        token_field: 回應 JSON 中 token 的欄位名稱，預設 "access_token"。
        expires_field: 回應 JSON 中過期秒數的欄位名稱，預設 "expires_in"。
        auth_method: 認證方式，"bearer"（Authorization: Bearer J1）或 "body"（放入 request body）。
        extra_body: 額外 POST body 欄位。
        extra_headers: 額外 request headers。
        buffer_seconds: 提前幾秒視為過期，預設 60 秒。
    """

    def __init__(
        self,
        auth_url: str,
        auth_token: str,
        *,
        token_field: str = "access_token",
        expires_field: str = "expires_in",
        auth_method: str = "bearer",
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        buffer_seconds: int = 60,
    ) -> None:
        self.auth_url = auth_url
        self.auth_token = auth_token
        self.token_field = token_field
        self.expires_field = expires_field
        self.auth_method = auth_method
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self.buffer_seconds = buffer_seconds

        self._cached_token: str | None = None
        self._expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        """token 是否已過期（含 buffer）。"""
        if self._cached_token is None:
            return True
        return time.time() >= (self._expires_at - self.buffer_seconds)

    def get_token(self) -> str:
        """取得有效的 J2 token（自動快取，過期自動重取）。

        Returns:
            J2 access token 字串。

        Raises:
            httpx.HTTPStatusError: 驗證請求失敗。
            KeyError: 回應中找不到 token 欄位。
        """
        if not self.is_expired:
            return self._cached_token  # type: ignore

        return self._exchange()

    def _exchange(self) -> str:
        """執行 token 交換。"""
        headers = {**self.extra_headers}

        if self.auth_method == "bearer":
            headers["Authorization"] = f"Bearer {self.auth_token}"
            body = {**self.extra_body}
        else:
            # auth_method == "body"
            body = {"token": self.auth_token, **self.extra_body}

        logger.debug(f"Exchanging token at {self.auth_url}")

        with httpx.Client(timeout=30) as client:
            resp = client.post(self.auth_url, json=body, headers=headers)
            resp.raise_for_status()

        data = resp.json()

        if self.token_field not in data:
            raise KeyError(
                f"Token field '{self.token_field}' not found in auth response. "
                f"Available keys: {list(data.keys())}"
            )

        self._cached_token = data[self.token_field]

        # 解析過期時間
        expires_in = data.get(self.expires_field)
        if expires_in:
            self._expires_at = time.time() + float(expires_in)
        else:
            # 無過期資訊，預設 1 小時
            self._expires_at = time.time() + 3600

        logger.info(
            f"Token exchanged successfully, "
            f"expires in {int(self._expires_at - time.time())}s"
        )

        return self._cached_token  # type: ignore

    def clear_cache(self) -> None:
        """清除快取的 token，下次 get_token() 會重新交換。"""
        self._cached_token = None
        self._expires_at = 0.0
