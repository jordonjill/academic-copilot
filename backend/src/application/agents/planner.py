from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.domain.state import GlobalState as GraphState
from src.application.agents.factory import build_planner
from src.infrastructure.config.config import MAX_SEARCHES


def planner_node(state: GraphState, llm: BaseLanguageModel) -> Dict:
    resources = state.get("retrieved_resources", [])
    resources_text = "\n\n---\n\n".join(
        [f"Source URI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}" for r in resources]
    )

    chain = build_planner(llm)
    research_plan = chain.invoke({
        "initial_topic": state["initial_topic"],
        "retrieved_resources": resources_text,
    })

    search_count = state.get("search_count", 0)
    if research_plan.step_type == "search" and search_count >= MAX_SEARCHES:
        print(f"\n[!] Search limit reached ({MAX_SEARCHES}). Forcing synthesize.")
        research_plan.step_type = "synthesize"
        research_plan.query = None

    return {"research_plan": research_plan}
