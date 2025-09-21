import os
import re
import requests
from typing import List, Dict, Any
from src.config.config import MAX_TAVILY_SEARCHES
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

class JinaClient:
    def crawl(self, url: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Retain-Images": "none"
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            print("Jina API key is not set. Using limited access.")

        data = {"url": url}
        response = requests.post("https://r.jina.ai/", headers=headers, json=data, timeout=120)
        return response.text

def extract_article_info(content: str) -> dict:
    results = {}

    title_pattern = re.compile(r"^Title: (.*)$", re.MULTILINE)
    title_match = title_pattern.search(content)
    results["title"] = title_match.group(1).strip() if title_match else "None"
    
    abstract_pattern = re.compile(r"^Abstract\n-+\s*([\s\S]+?)(?=\n\s*\n)", re.MULTILINE)
    abstract_match = abstract_pattern.search(content)
    abstract = "None"

    if abstract_match:
        abstract = abstract_match.group(1).strip()

    if abstract != "None":
        abstract = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', abstract)
    
    results["abstract"] = abstract

    return results

def crawler_tool(url: str) -> Dict[str, str]:
    try:
        crawler = JinaClient()
        article = crawler.crawl(url)
        content = extract_article_info(article)
        
        result = {"uri": url,
                  "title": content.get('title', 'None'),
                  "content": content.get('abstract', 'None')
                  }

        return result

    except Exception as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        return error_msg

@tool
def crawl_search(query: str) -> List[Dict[str, Any]]:
    """
    This tool is used to search the web for information based on a query term.
    """
    try:
        tavily_search = TavilySearch(
            max_results=MAX_TAVILY_SEARCHES,
            search_depth="advanced",
            include_domains=["https://www.sciencedirect.com/"]
        )

        results = tavily_search.invoke({"query": query})

        urls = [res.get("url") for res in results['results'] if res.get("url")]

        with ThreadPoolExecutor(max_workers=5) as executor:
            content = list(executor.map(crawler_tool, urls))

        return content
    
    except Exception as e:
        return [f"Search failed. Error: {repr(e)}"]
