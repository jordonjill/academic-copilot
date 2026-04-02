from __future__ import annotations

from concurrent.futures import Future

from src.infrastructure.tools import academic_tools


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            future.set_exception(exc)
        return future


def test_scholar_search_degrades_to_web_when_arxiv_fails_and_include_web_disabled(monkeypatch):
    monkeypatch.setattr(academic_tools, "_SCHOLAR_EXECUTOR", _ImmediateExecutor())

    def _arxiv_fail(*args, **kwargs):
        raise RuntimeError("arxiv unavailable")

    web_calls = {"count": 0}

    def _web_ok(query: str, limit: int):
        web_calls["count"] += 1
        return [
            {
                "title": "web hit",
                "uri": "https://example.com/paper",
                "summary": "ok",
                "venue": "web",
                "year": "",
                "source_type": "web",
            }
        ]

    monkeypatch.setattr(academic_tools, "_arxiv_search", _arxiv_fail)
    monkeypatch.setattr(academic_tools, "_web_search", _web_ok)

    rows = academic_tools.scholar_search.func("test query", max_results=5, include_web=False)
    assert isinstance(rows, list)
    assert rows and rows[0].get("source_type") == "web"
    assert web_calls["count"] == 1


def test_scholar_search_retries_web_only_when_parallel_web_empty(monkeypatch):
    monkeypatch.setattr(academic_tools, "_SCHOLAR_EXECUTOR", _ImmediateExecutor())

    def _arxiv_fail(*args, **kwargs):
        raise RuntimeError("arxiv failed")

    web_calls = {"count": 0}

    def _web_retry(query: str, limit: int):
        web_calls["count"] += 1
        if web_calls["count"] == 1:
            return []
        return [
            {
                "title": "fallback web hit",
                "uri": "https://example.com/fallback",
                "summary": "ok",
                "venue": "web",
                "year": "",
                "source_type": "web",
            }
        ]

    monkeypatch.setattr(academic_tools, "_arxiv_search", _arxiv_fail)
    monkeypatch.setattr(academic_tools, "_web_search", _web_retry)

    rows = academic_tools.scholar_search.func("test query", max_results=5, include_web=True)
    assert isinstance(rows, list)
    assert rows and rows[0].get("uri") == "https://example.com/fallback"
    assert web_calls["count"] == 2

