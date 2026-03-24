from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.domain.state import GlobalState as GraphState
from src.application.agents.factory import build_synthesizer


def synthesizer_node(state: GraphState, llm: BaseLanguageModel) -> Dict:
    plan = state.get("research_plan")
    if not plan or plan.step_type != "synthesize":
        raise ValueError("Synthesizer called outside of synthesize step.")
    if not state.get("retrieved_resources"):
        raise ValueError("Synthesizer requires retrieved resources.")

    resources_text = "\n\n---\n\n".join(
        [f"Source URI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}"
         for r in state["retrieved_resources"]]
    )

    feedback_section = ""
    if (critic := state.get("research_critic")) and critic.feedback:
        prev_idea = state["research_creation"].research_idea
        feedback_section = (
            f"\n**Previous Rejected Idea:** {prev_idea}"
            f"\n**Critic Feedback:** {critic.feedback}"
            f"\nGenerate a NEW idea that addresses this feedback."
        )

    chain = build_synthesizer(llm)
    research_creation = chain.invoke({
        "initial_topic": state["initial_topic"],
        "retrieved_resources": resources_text,
        "feedback_section": feedback_section,
    })
    return {"research_creation": research_creation}
