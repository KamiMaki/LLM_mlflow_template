"""app.api 單元測試。"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import Depends
from fastapi.testclient import TestClient

from app.main import create_app
from app.api.auth import require_auth


class TestAPI:
    def test_health(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestBearerAuth:
    """Bearer token 認證測試。"""

    def _app_with_protected_route(self):
        """建立含有受保護路由的 app。"""
        from fastapi import APIRouter

        app = create_app()
        router = APIRouter()

        @router.get("/protected")
        async def protected(token: str = Depends(require_auth)):
            return {"message": "ok", "token": token}

        app.include_router(router)
        return app

    def test_no_auth_required_when_token_empty(self):
        """auth_token 未設定時，不需驗證。"""
        app = self._app_with_protected_route()
        client = TestClient(app)

        with patch("app.api.auth._get_auth_token", return_value=""):
            resp = client.get("/protected")
            assert resp.status_code == 200

    def test_valid_token(self):
        """正確 token 通過驗證。"""
        app = self._app_with_protected_route()
        client = TestClient(app)

        with patch("app.api.auth._get_auth_token", return_value="secret-key"):
            resp = client.get(
                "/protected",
                headers={"Authorization": "Bearer secret-key"},
            )
            assert resp.status_code == 200
            assert resp.json()["message"] == "ok"

    def test_missing_token_returns_401(self):
        """缺少 token 時回傳 401。"""
        app = self._app_with_protected_route()
        client = TestClient(app)

        with patch("app.api.auth._get_auth_token", return_value="secret-key"):
            resp = client.get("/protected")
            assert resp.status_code == 401

    def test_wrong_token_returns_401(self):
        """錯誤 token 回傳 401。"""
        app = self._app_with_protected_route()
        client = TestClient(app)

        with patch("app.api.auth._get_auth_token", return_value="secret-key"):
            resp = client.get(
                "/protected",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401
