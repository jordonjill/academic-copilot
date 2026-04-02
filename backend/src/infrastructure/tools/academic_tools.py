from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool


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


@tool
def scholar_search(query: str, max_results: int = 12, include_web: bool = True) -> list[dict[str, Any]]:
    """Unified academic search over arXiv and optional web fallback."""
    limit = max(1, min(int(max_results), 30))
    merged: list[dict[str, Any]] = []
    seen_uri: set[str] = set()

    # arXiv primary search
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for paper in client.results(search):
            uri = str(getattr(paper, "entry_id", "") or "").strip()
            if not uri or uri in seen_uri:
                continue
            seen_uri.add(uri)
            merged.append(
                {
                    "title": str(getattr(paper, "title", "") or "").strip(),
                    "uri": uri,
                    "summary": str(getattr(paper, "summary", "") or "")[:1200],
                    "venue": "arXiv",
                    "year": str(getattr(getattr(paper, "published", None), "year", "") or ""),
                    "source_type": "paper",
                }
            )
            if len(merged) >= limit:
                break
    except Exception:
        # Keep silent and try web fallback.
        pass

    if include_web and len(merged) < limit:
        try:
            from langchain_tavily import TavilySearch

            web_limit = max(1, min(limit - len(merged), limit))
            tavily = TavilySearch(max_results=web_limit, search_depth="advanced")
            results = tavily.invoke({"query": query})
            items = results.get("results") if isinstance(results, dict) else []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    uri = str(item.get("url") or "").strip()
                    if not uri or uri in seen_uri:
                        continue
                    seen_uri.add(uri)
                    merged.append(
                        {
                            "title": str(item.get("title") or "").strip(),
                            "uri": uri,
                            "summary": str(item.get("content") or "")[:800],
                            "venue": "web",
                            "year": "",
                            "source_type": "web",
                        }
                    )
                    if len(merged) >= limit:
                        break
        except Exception:
            pass

    if merged:
        return merged
    return [
        _tool_error(
            "SCHOLAR_SEARCH_EMPTY",
            "No results returned. Check query quality or API/dependency availability.",
        )
    ]


@tool
def paper_fetch(uri: str, max_chars: int = 5000, timeout_seconds: int = 20) -> dict[str, Any]:
    """Fetch abstract/full text preview from a paper/page URI."""
    target = (uri or "").strip()
    if not target:
        return _tool_error("PAPER_FETCH_INVALID_URI", "uri is required")

    max_chars = max(200, min(int(max_chars), 20000))
    timeout = max(3, min(int(timeout_seconds), 60))

    if "arxiv.org" in target:
        arxiv_id_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", target)
        if arxiv_id_match:
            arxiv_id = arxiv_id_match.group(1).replace(".pdf", "")
            try:
                import arxiv

                search = arxiv.Search(id_list=[arxiv_id], max_results=1)
                paper = next(arxiv.Client().results(search), None)
                if paper is not None:
                    summary = str(getattr(paper, "summary", "") or "")
                    return {
                        "ok": True,
                        "uri": target,
                        "title": str(getattr(paper, "title", "") or ""),
                        "content": summary[:max_chars],
                        "truncated": len(summary) > max_chars,
                    }
            except Exception:
                pass

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
        "checked_at": datetime.utcnow().isoformat() + "Z",
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
