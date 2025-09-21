from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.state import GraphState, ResearchPlan
from src.config.prompt import PLANNER_PROMPT
from src.config.config import MAX_SEARCHES

def planner_node(state: GraphState, llm: BaseLanguageModel) -> Dict:

    resources = state.get("retrieved_resources", [])
    resources = "\n\n---\n\n".join(
        [f"Source URI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}" for r in resources]
    )

    planner_chain = PLANNER_PROMPT | llm.with_structured_output(ResearchPlan)
    research_plan = planner_chain.invoke({
        "initial_topic": state["initial_topic"],
        "retrieved_resources": resources
    })

    search_count = state.get("search_count", 0)
    
    if research_plan.step_type == "search":
        if search_count >= MAX_SEARCHES:
            print(f"\n[!] Warning: Search count has reached the limit of {MAX_SEARCHES}.")
            research_plan.step_type = "synthesize"
            research_plan.query = None

    return {"research_plan": research_plan}
