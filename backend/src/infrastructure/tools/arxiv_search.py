"""
ArXiv 搜索工具（使用 arxiv Python 库）。

依赖：pip install arxiv
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search ArXiv for academic papers matching the query.

    Returns a list of paper metadata dicts with keys:
      uri, title, content (abstract), authors, published
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=min(max_results, 20),
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append({
                "uri": paper.entry_id,
                "title": paper.title,
                "content": paper.summary[:800],
                "authors": ", ".join(str(a) for a in paper.authors[:5]),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })
        return results

    except ImportError:
        return [{"error_code": "ARXIV_NOT_INSTALLED", "error_message": "arxiv package not installed. Run: pip install arxiv"}]
    except Exception as e:
        logger.exception("ArXiv search failed for query=%s", query)
        return [{"error_code": "ARXIV_SEARCH_FAILED", "error_message": repr(e)}]
