from typing import Dict
from src.state import GlobalState, Resource
from src.tools import crawl_search

def researcher_node(state: GlobalState) -> Dict:

    plan = state["research_plan"]
    if not plan or plan.step_type != "search":
        raise ValueError("Researcher node should only be called when the plan is to search.")

    query = plan.query
    raw_results = crawl_search.invoke({"query": query})
    new_resources = []
    for result in raw_results:
        if isinstance(result, dict) and "error" not in result:
            try:
                new_resources.append(Resource(**result))
            except Exception as e:
                print(f"[!] Warning: Could not create Resource object from data: {result}. Error: {e}")

    # operator.add reducer: 仅返回新增资源（reducer 自动追加到现有列表）
    search_count = state.get("search_count", 0) + 1

    return {
        "retrieved_resources": new_resources,
        "search_count": search_count,
    }