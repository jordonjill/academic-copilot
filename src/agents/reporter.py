from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.state import GlobalState as GraphState
from src.agents.factory import build_reporter


def reporter_node(state: GraphState, llm: BaseLanguageModel) -> Dict:
    critic = state.get("research_critic")
    if not state.get("research_creation") or not critic or not critic.is_valid:
        raise ValueError("Reporter called before idea validation.")

    creation = state["research_creation"]
    resources = state["retrieved_resources"]
    all_resources_text = "\n\n---\n\n".join(
        [f"Source {i+1}:\nURI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}"
         for i, r in enumerate(resources)]
    )

    chain = build_reporter(llm)
    final_proposal = chain.invoke({
        "initial_topic": state["initial_topic"],
        "research_gap": creation.research_gap,
        "research_idea": creation.research_idea,
        "feedback_section": critic.feedback or "",
        "all_resources": all_resources_text,
    })

    # 覆盖参考文献确保准确
    final_proposal.References = [{"title": r.title, "uri": r.uri} for r in resources]
    return {"final_proposal": final_proposal}
