from __future__ import annotations

from src.interfaces.api import service
from src.interfaces.api.service import AcademicCopilotApp, create_copilot


def test_health_check_returns_subprobes(monkeypatch):
    app = AcademicCopilotApp()
    monkeypatch.setattr(app.runtime, "health_probe", lambda: {"ok": True, "loaded_agents": 3})
    monkeypatch.setattr(service, "_memory_probe", lambda: {"ok": True})

    result = app.health_check()
    assert result["status"] == "healthy"
    assert "runtime" in result["probe"]
    assert "memory" in result["probe"]
    assert "config" in result["probe"]
    assert result["probe"]["memory"]["ok"] is True


def test_health_check_marks_unhealthy_when_memory_probe_fails(monkeypatch):
    app = AcademicCopilotApp()
    monkeypatch.setattr(app.runtime, "health_probe", lambda: {"ok": True, "loaded_agents": 3})
    monkeypatch.setattr(service, "_memory_probe", lambda: {"ok": False, "error": "db down"})

    result = app.health_check()
    assert result["status"] == "unhealthy"
    assert result["probe"]["memory"]["ok"] is False


def test_create_copilot_reuses_singleton_instance(monkeypatch):
    monkeypatch.setattr(service, "_COPILOT_APP", None)
    app1 = create_copilot()
    app2 = create_copilot()
    assert app1 is app2
