from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_ARXIV_COOLDOWN_LOCK = threading.Lock()
_ARXIV_COOLDOWN_UNTIL = 0.0
_SCHOLAR_MAX_WORKERS = 4
_SCHOLAR_SOURCE_TIMEOUT_SECONDS = 8.0
_ARXIV_CLIENT_NUM_RETRIES = 0
_ARXIV_CLIENT_DELAY_SECONDS = 0.2
_ARXIV_CLIENT_PAGE_SIZE = 8
_ARXIV_RATE_LIMIT_COOLDOWN_SECONDS = 300.0
_PAPER_FETCH_DEFAULT_TIMEOUT_SECONDS = 10


_SCHOLAR_EXECUTOR = ThreadPoolExecutor(
    max_workers=_SCHOLAR_MAX_WORKERS,
    thread_name_prefix="scholar-search",
)


def _shutdown_scholar_executor() -> None:
    _SCHOLAR_EXECUTOR.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_scholar_executor)


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": code,
        "error_message": message,
    }


def _resolve_root() -> Path:
    raw = os.getenv("LOCAL_DOC_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _safe_candidate(root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    candidate.relative_to(root)
    return candidate


def _strip_html(text: str) -> str:
    no_script = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    no_style = re.sub(r"<style[\s\S]*?</style>", "", no_script, flags=re.I)
    no_tags = re.sub(r"<[^>]+>", " ", no_style)
    compact = re.sub(r"\s+", " ", no_tags).strip()
    return compact


def _normalize_uris(seed_uris: list[str] | str) -> list[str]:
    if isinstance(seed_uris, str):
        raw_items = re.split(r"[\n,]", seed_uris)
    elif isinstance(seed_uris, list):
        raw_items = [item for item in seed_uris if isinstance(item, str)]
    else:
        raw_items = []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        uri = item.strip()
        if not uri:
            continue
        if uri in seen:
            continue
        seen.add(uri)
        result.append(uri)
    return result


def _arxiv_rate_limited() -> bool:
    with _ARXIV_COOLDOWN_LOCK:
        return time.monotonic() < _ARXIV_COOLDOWN_UNTIL


def _mark_arxiv_cooldown() -> None:
    with _ARXIV_COOLDOWN_LOCK:
        global _ARXIV_COOLDOWN_UNTIL
        _ARXIV_COOLDOWN_UNTIL = time.monotonic() + _ARXIV_RATE_LIMIT_COOLDOWN_SECONDS


def _is_arxiv_rate_limit_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _dedupe_and_take(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_uri: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("uri") or "").strip()
        if not uri or uri in seen_uri:
            continue
        seen_uri.add(uri)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _extract_arxiv_id(target: str) -> str:
    text = (target or "").strip()
    if not text:
        return ""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", text, flags=re.I)
    if not m:
        return ""
    return m.group(1).replace(".pdf", "").strip()


def _fetch_arxiv_entry_via_api(arxiv_id: str, timeout_seconds: int, max_chars: int) -> dict[str, Any]:
    if _arxiv_rate_limited():
        return _tool_error(
            "PAPER_FETCH_ARXIV_COOLDOWN",
            "arXiv access is in cooldown due to rate limit; skip for now and continue with other sources.",
        )

    api_url = "https://export.arxiv.org/api/query"
    try:
        response = requests.get(
            api_url,
            params={"id_list": arxiv_id, "max_results": 1},
            timeout=max(3, timeout_seconds),
        )
        if response.status_code == 429:
            _mark_arxiv_cooldown()
            return _tool_error("PAPER_FETCH_ARXIV_RATE_LIMITED", "arXiv returned HTTP 429")
        response.raise_for_status()
    except Exception as exc:
        if _is_arxiv_rate_limit_error(exc):
            _mark_arxiv_cooldown()
            return _tool_error("PAPER_FETCH_ARXIV_RATE_LIMITED", repr(exc))
        return _tool_error("PAPER_FETCH_ARXIV_FAILED", repr(exc))

    try:
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return _tool_error("PAPER_FETCH_ARXIV_EMPTY", f"arXiv id not found: {arxiv_id}")

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        summary = re.sub(r"\s+", " ", summary).strip()
        return {
            "ok": True,
            "uri": f"https://arxiv.org/abs/{arxiv_id}",
            "title": title,
            "content": summary[:max_chars],
            "truncated": len(summary) > max_chars,
        }
    except Exception as exc:
        return _tool_error("PAPER_FETCH_ARXIV_PARSE_FAILED", repr(exc))


def _arxiv_search(query: str, limit: int) -> list[dict[str, Any]]:
    if _arxiv_rate_limited():
        logger.warning("scholar_search.arxiv_cooldown_active skip=true")
        return []

    try:
        import arxiv

        client = arxiv.Client(
            page_size=min(_ARXIV_CLIENT_PAGE_SIZE, limit),
            delay_seconds=_ARXIV_CLIENT_DELAY_SECONDS,
            num_retries=_ARXIV_CLIENT_NUM_RETRIES,
        )
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        rows: list[dict[str, Any]] = []
        for paper in client.results(search):
            rows.append(
                {
                    "title": str(getattr(paper, "title", "") or "").strip(),
                    "uri": str(getattr(paper, "entry_id", "") or "").strip(),
                    "summary": str(getattr(paper, "summary", "") or "")[:1200],
                    "venue": "arXiv",
                    "year": str(getattr(getattr(paper, "published", None), "year", "") or ""),
                    "source_type": "paper",
                }
            )
            if len(rows) >= limit:
                break
        return _dedupe_and_take(rows, limit)
    except Exception as exc:
        if _is_arxiv_rate_limit_error(exc):
            _mark_arxiv_cooldown()
            logger.warning("scholar_search.arxiv_rate_limited cooldown_applied=true error=%s", exc)
        else:
            logger.warning("scholar_search.arxiv_failed error=%s", exc)
        return []


def _web_search(query: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        from langchain_tavily import TavilySearch

        tavily = TavilySearch(max_results=limit, search_depth="advanced")
        results = tavily.invoke({"query": query})
        items = results.get("results") if isinstance(results, dict) else []
        rows: list[dict[str, Any]] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "title": str(item.get("title") or "").strip(),
                        "uri": str(item.get("url") or "").strip(),
                        "summary": str(item.get("content") or "")[:800],
                        "venue": "web",
                        "year": "",
                        "source_type": "web",
                    }
                )
                if len(rows) >= limit:
                    break
        return _dedupe_and_take(rows, limit)
    except Exception as exc:
        logger.warning("scholar_search.web_failed error=%s", exc)
        return []


def _future_result_or_empty(
    future: Future[list[dict[str, Any]]],
    *,
    source_name: str,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], str]:
    try:
        return future.result(timeout=timeout_seconds), "ok"
    except FutureTimeoutError:
        future.cancel()
        logger.warning(
            "scholar_search.source_timeout source=%s timeout_seconds=%.2f",
            source_name,
            timeout_seconds,
        )
        return [], "timeout"
    except Exception as exc:
        logger.warning("scholar_search.source_failed source=%s error=%s", source_name, exc)
        return [], "failed"


@tool
def scholar_search(query: str, max_results: int = 12, include_web: bool = True) -> list[dict[str, Any]]:
    """Unified academic search over arXiv + optional web fallback (parallel, timeout-aware)."""
    limit = max(1, min(int(max_results), 20))
    source_timeout = _SCHOLAR_SOURCE_TIMEOUT_SECONDS
    started = time.monotonic()
    logger.info(
        "scholar_search.start include_web=%s max_results=%s query_preview=%s",
        include_web,
        limit,
        (query or "")[:120],
    )

    arxiv_future: Future[list[dict[str, Any]]] = _SCHOLAR_EXECUTOR.submit(_arxiv_search, query, limit)
    web_future: Future[list[dict[str, Any]]] | None = None
    if include_web:
        web_future = _SCHOLAR_EXECUTOR.submit(_web_search, query, limit)

    arxiv_rows, arxiv_status = _future_result_or_empty(
        arxiv_future,
        source_name="arxiv",
        timeout_seconds=source_timeout,
    )
    web_rows: list[dict[str, Any]] = []
    web_status = "skipped"
    if web_future is not None:
        elapsed = max(0.0, time.monotonic() - started)
        remaining_timeout = max(0.0, source_timeout - elapsed)
        web_rows, web_status = _future_result_or_empty(
            web_future,
            source_name="web",
            timeout_seconds=remaining_timeout,
        )

    arxiv_unavailable = arxiv_status in {"timeout", "failed"} or _arxiv_rate_limited()
    if arxiv_unavailable and not web_rows:
        logger.warning(
            "scholar_search.degrade_web_only arxiv_status=%s include_web=%s",
            arxiv_status,
            include_web,
        )
        web_rows = _web_search(query, limit)
        web_status = "fallback"

    merged = _dedupe_and_take([*arxiv_rows, *web_rows], limit)
    logger.info(
        "scholar_search.done arxiv=%s web=%s merged=%s arxiv_status=%s web_status=%s elapsed=%.2fs",
        len(arxiv_rows),
        len(web_rows),
        len(merged),
        arxiv_status,
        web_status,
        max(0.0, time.monotonic() - started),
    )
    if merged:
        return merged
    return [
        _tool_error(
            "SCHOLAR_SEARCH_EMPTY",
            "No results returned. Check query quality or API/dependency availability.",
        )
    ]


@tool
def paper_fetch(
    uri: str,
    max_chars: int = 5000,
    timeout_seconds: int = _PAPER_FETCH_DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch abstract/full text preview from a paper/page URI."""
    target = (uri or "").strip()
    if not target:
        return _tool_error("PAPER_FETCH_INVALID_URI", "uri is required")

    max_chars = max(200, min(int(max_chars), 20000))
    timeout = max(3, min(int(timeout_seconds), 60))

    arxiv_id = _extract_arxiv_id(target)
    if arxiv_id:
        arxiv_result = _fetch_arxiv_entry_via_api(arxiv_id, timeout, max_chars)
        if bool(arxiv_result.get("ok")):
            return arxiv_result

    try:
        response = requests.get(target, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw_text = response.text
        clean_text = _strip_html(raw_text) if "html" in content_type.lower() else raw_text
        clean_text = clean_text.strip()
        return {
            "ok": True,
            "uri": target,
            "title": "",
            "content": clean_text[:max_chars],
            "truncated": len(clean_text) > max_chars,
        }
    except Exception as exc:
        return _tool_error("PAPER_FETCH_FAILED", repr(exc))


@tool
def pdf_structured_extract(path: str, max_chars: int = 7000) -> dict[str, Any]:
    """Extract compact structured information from a local PDF under LOCAL_DOC_ROOT."""
    root = _resolve_root()
    try:
        target = _safe_candidate(root, path)
    except Exception:
        return _tool_error("PDF_PATH_INVALID", "path must be within LOCAL_DOC_ROOT")

    if not target.exists() or not target.is_file():
        return _tool_error("PDF_NOT_FOUND", f"file not found: {target}")

    max_chars = max(500, min(int(max_chars), 30000))

    try:
        from pypdf import PdfReader
    except Exception:
        return _tool_error("PDF_DEPENDENCY_MISSING", "pypdf is required")

    try:
        reader = PdfReader(str(target))
        parts: list[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
            if sum(len(x) for x in parts) >= max_chars:
                break
        merged = "\n".join(parts)
        merged = merged[:max_chars]

        lines = [line.strip() for line in merged.splitlines() if line.strip()]
        headings: list[str] = []
        for line in lines[:120]:
            if len(line) > 120:
                continue
            if re.match(r"^(abstract|introduction|method|methods|experiment|results|conclusion)\b", line, re.I):
                headings.append(line)
            elif line.isupper() and 3 <= len(line) <= 80:
                headings.append(line)
            if len(headings) >= 10:
                break

        return {
            "ok": True,
            "path": str(target.relative_to(root)),
            "chars": len(merged),
            "headings": headings,
            "content": merged,
        }
    except Exception as exc:
        return _tool_error("PDF_EXTRACT_FAILED", repr(exc))


@tool
def citation_graph(seed_uris: list[str] | str, max_nodes: int = 20) -> dict[str, Any]:
    """Build a lightweight citation graph skeleton from known URIs."""
    uris = _normalize_uris(seed_uris)
    max_nodes = max(1, min(int(max_nodes), 200))
    uris = uris[:max_nodes]

    nodes = [{"id": uri, "uri": uri} for uri in uris]
    edges: list[dict[str, str]] = []
    for idx in range(1, len(uris)):
        edges.append({"from": uris[idx - 1], "to": uris[idx], "type": "related"})

    return {
        "ok": True,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


@tool
def claim_grounding_check(
    claims: list[dict[str, Any]] | list[str] | str,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check whether claims are grounded by citation or support URI metadata."""
    normalized_claims: list[dict[str, Any]] = []

    if isinstance(claims, str):
        claim_lines = [line.strip() for line in claims.splitlines() if line.strip()]
        for line in claim_lines:
            normalized_claims.append({"claim": line})
    elif isinstance(claims, list):
        for item in claims:
            if isinstance(item, str) and item.strip():
                normalized_claims.append({"claim": item.strip()})
            elif isinstance(item, dict):
                normalized_claims.append(item)

    citation_uris: set[str] = set()
    if isinstance(citations, list):
        for item in citations:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            if uri:
                citation_uris.add(uri)

    grounded = 0
    details: list[dict[str, Any]] = []
    for item in normalized_claims:
        support_uri = str(item.get("support_uri") or item.get("uri") or "").strip()
        claim_text = str(item.get("claim") or "").strip()
        ok = bool(support_uri) and (not citation_uris or support_uri in citation_uris)
        if ok:
            grounded += 1
        details.append(
            {
                "claim": claim_text,
                "support_uri": support_uri,
                "grounded": ok,
            }
        )

    total = len(normalized_claims)
    score = 1.0 if total == 0 else grounded / float(total)
    return {
        "ok": True,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "grounded_count": grounded,
        "total_claims": total,
        "grounding_score": round(score, 4),
        "details": details,
    }


@tool
def bib_manager(citations: list[dict[str, Any]], style: str = "bibtex") -> dict[str, Any]:
    """Generate citation export entries from normalized citation records."""
    if style.lower() != "bibtex":
        return _tool_error("BIB_STYLE_UNSUPPORTED", "currently only style='bibtex' is supported")

    entries: list[str] = []
    for idx, item in enumerate(citations or [], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Untitled").replace("{", "").replace("}", "")
        uri = str(item.get("uri") or "")
        year = str(item.get("year") or "")
        key = f"ref{idx}"
        entry = (
            f"@misc{{{key},\\n"
            f"  title={{ {title} }},\\n"
            f"  howpublished={{\\url{{{uri}}}}},\\n"
            f"  year={{ {year} }}\\n"
            f"}}"
        )
        entries.append(entry)

    return {
        "ok": True,
        "style": "bibtex",
        "count": len(entries),
        "entries": entries,
        "content": "\\n\\n".join(entries),
    }
