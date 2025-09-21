from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.state import GraphState, FinalProposal
from src.config.prompt import REPORTER_PROMPT

def reporter_node(state: GraphState, llm: BaseLanguageModel) -> Dict:
    
    research_critic = state.get("research_critic")
    if not state.get("research_creation") or not research_critic or not research_critic.is_valid:
        raise ValueError("Reporter node should only be called after a research idea has been validated.")

    initial_topic = state["initial_topic"]
    research_creation = state["research_creation"]
    research_gap = research_creation.research_gap
    research_idea = research_creation.research_idea
    original_resources = state["retrieved_resources"]
    research_critic = state["research_critic"]
    feedback_section = research_critic.feedback
    all_resources = "\n\n---\n\n".join(
        [f"Source {i+1}:\nURI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}" for i, r in enumerate(original_resources)]
    )

    reporter_chain = REPORTER_PROMPT | llm.with_structured_output(FinalProposal)
    
    final_proposal = reporter_chain.invoke({
        "initial_topic": initial_topic,
        "research_gap": research_gap,
        "research_idea": research_idea,
        "feedback_section": feedback_section,
        "all_resources": all_resources
    })

    # Always override with correct references from resources
    final_proposal.References = [{"title": r.title, "uri": r.uri} for r in original_resources]

    return {
        "final_proposal": final_proposal,
    }
