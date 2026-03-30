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


def test_admin_reload_returns_200_with_runtime_and_tools(monkeypatch):
    report = {
        "runtime": {
            "config_version": 3,
            "loaded": {"agents": ["planner"], "workflows": ["proposal_v2"]},
            "failed": [],
        },
        "tools": {
            "version": 2,
            "loaded_tools": ["arxiv", "web_search"],
            "loaded_servers": [],
            "tool_catalog_path": "/tmp/tools.yaml",
        },
    }

    async def _reload_all():
        return report

    monkeypatch.setattr("src.interfaces.api.routes.admin.reload_all_config", _reload_all)

    client = _build_client()
    response = client.post("/admin/reload", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["runtime"]["config_version"] == 3
    assert payload["data"]["tools"]["version"] == 2


def test_admin_reload_tools_returns_200(monkeypatch):
    report = {
        "version": 7,
        "loaded_tools": ["arxiv", "web_search", "filesystem"],
        "loaded_servers": ["filesystem"],
        "tool_catalog_path": "/tmp/tools.yaml",
    }

    async def _reload_tools():
        return report

    monkeypatch.setattr("src.interfaces.api.routes.admin.reload_tools_config", _reload_tools)

    client = _build_client()
    response = client.post("/admin/reload-tools", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["version"] == 7


def test_chat_accepts_workflow_id_and_returns_200(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "src.interfaces.api.routes.chat.create_copilot",
        lambda: _DummyCopilot(captured),
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
