from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.infrastructure.config.config import MAX_TAVILY_SEARCHES

load_dotenv()
logger = logging.getLogger(__name__)

_JINA_TIMEOUT_SECONDS = 120
_CRAWL_MAX_WORKERS = 5
_DEFAULT_DOMAIN_FILTER = ["https://www.sciencedirect.com/"]


def _tool_error(code: str, message: str, *, uri: str | None = None) -> dict[str, str]:
    payload: dict[str, str] = {
        "error_code": code,
        "error_message": message,
    }
    if uri:
        payload["uri"] = uri
    return payload


def _resolve_domain_filter() -> list[str]:
    raw = os.getenv("WEB_SEARCH_INCLUDE_DOMAINS", "").strip()
    if not raw:
        return list(_DEFAULT_DOMAIN_FILTER)
    return [item.strip() for item in raw.split(",") if item.strip()]


class JinaClient:
    def crawl(self, url: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Retain-Images": "none",
        }
        jina_api_key = os.getenv("JINA_API_KEY")
        if jina_api_key:
            headers["Authorization"] = f"Bearer {jina_api_key}"
        else:
            logger.debug("Jina API key not set. Using limited access mode.")

        response = requests.post(
            "https://r.jina.ai/",
            headers=headers,
            json={"url": url},
            timeout=_JINA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.text


def extract_article_info(content: str) -> dict[str, str]:
    results: dict[str, str] = {}

    title_pattern = re.compile(r"^Title: (.*)$", re.MULTILINE)
    title_match = title_pattern.search(content)
    results["title"] = title_match.group(1).strip() if title_match else "None"

    abstract_pattern = re.compile(r"^Abstract\n-+\s*([\s\S]+?)(?=\n\s*\n)", re.MULTILINE)
    abstract_match = abstract_pattern.search(content)
    abstract = "None"
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    if abstract != "None":
        abstract = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", abstract)
    results["abstract"] = abstract
    return results


def crawler_tool(url: str) -> dict[str, str]:
    try:
        crawler = JinaClient()
        article = crawler.crawl(url)
        content = extract_article_info(article)
        return {
            "uri": url,
            "title": content.get("title", "None"),
            "content": content.get("abstract", "None"),
        }
    except Exception as exc:
        logger.warning("Crawler failed for url=%s: %s", url, exc)
        return _tool_error("CRAWL_FAILED", repr(exc), uri=url)


@tool
def crawl_search(query: str) -> list[dict[str, Any]]:
    """
    Search the web and extract compact article snippets.
    """
    try:
        tavily_search = TavilySearch(
            max_results=MAX_TAVILY_SEARCHES,
            search_depth="advanced",
            include_domains=_resolve_domain_filter(),
        )
        results = tavily_search.invoke({"query": query})
        items = results.get("results") if isinstance(results, dict) else []
        if not isinstance(items, list):
            return [_tool_error("SEARCH_BAD_RESPONSE", "Tavily response is not a list.")]

        urls: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url and url not in seen:
                seen.add(url)
                urls.append(url)

        if not urls:
            return []

        max_workers = min(_CRAWL_MAX_WORKERS, len(urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(crawler_tool, urls))
    except Exception as exc:
        logger.exception("Web search failed for query=%s", query)
        return [_tool_error("SEARCH_FAILED", repr(exc))]
