from typing import Dict
from src.state import GraphState, Resource
from src.tools import crawl_search

def researcher_node(state: GraphState) -> Dict:

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


    existing_resources = state.get("retrieved_resources", [])
    all_resources = existing_resources + new_resources
    search_count = state.get("search_count", 0) + 1

    return {
        "retrieved_resources": all_resources,
        "search_count": search_count,
    }