from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from src.state import GraphState, ResearchCreation
from src.config.prompt import SYNTHESIZER_PROMPT

def synthesizer_node(state: GraphState, llm: BaseLanguageModel) -> Dict:
    
    plan = state.get("research_plan")
    if not plan or plan.step_type != "synthesize":
        raise ValueError("Synthesizer node should only be called when the plan is to synthesize.")

    if not state.get("retrieved_resources"):
        raise ValueError("Synthesizer node cannot be called without retrieved resources.")

    resources = state["retrieved_resources"]
    resources = "\n\n---\n\n".join(
        [f"Source URI: {r.uri}\nTitle: {r.title}\nAbstract: {r.content}" for r in resources]
    )
    initial_topic = state["initial_topic"]

    feedback_section = ""
    if (research_critic := state.get("research_critic")) and research_critic.feedback:
        previous_idea = state["research_creation"].research_idea
        feedback_section = f"""
                            **Previous Attempt Feedback:**
                            You previously generated an idea that was reviewed by a Critic. You MUST generate a NEW and DIFFERENT idea that addresses the Critic's feedback.
                            - Previous Rejected Idea: '{previous_idea}'
                            - Critic's Feedback: '{research_critic.feedback}'
                            """
        

    synthesizer_chain = SYNTHESIZER_PROMPT | llm.with_structured_output(ResearchCreation)

    research_creation = synthesizer_chain.invoke({
        "initial_topic": initial_topic,
        "retrieved_resources": resources,
        "feedback_section": feedback_section
    })

    return {
        "research_creation": research_creation,
    }

