from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.interfaces.api.routes import admin, chat


AUTH_HEADERS = {"Authorization": "Bearer 123"}


class _DummyCopilot:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def chat_async(self, **kwargs):
        self._captured.update(kwargs)
        return {"success": True, "type": "chat", "message": "ok"}


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(admin.router)
    return TestClient(app)


def test_admin_reload_returns_200_and_config_version(monkeypatch):
    report = {
        "config_version": 3,
        "loaded": {"agents": ["planner"], "workflows": ["proposal_v2"]},
        "failed": [],
    }
    monkeypatch.setattr(
        "src.interfaces.api.routes.admin.reload_runtime_config",
        lambda: report,
    )

    client = _build_client()
    response = client.post("/admin/reload", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["config_version"] == 3


def test_chat_accepts_workflow_id_and_returns_200(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "src.interfaces.api.routes.chat.create_copilot",
        lambda model_type: _DummyCopilot(captured),
    )

    client = _build_client()
    response = client.post(
        "/chat",
        headers=AUTH_HEADERS,
        json={
            "message": "Hello",
            "workflow_id": "proposal_v2",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert captured["workflow_id"] == "proposal_v2"
