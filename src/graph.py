from functools import partial
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from src.state import GraphState
from src.agents import planner_node
from src.agents import researcher_node
from src.agents import synthesizer_node
from src.agents import critic_node
from src.agents import reporter_node
load_dotenv() 

llm = ChatOllama(model="llama3.1:8b", temperature=0)
# llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

planner = partial(planner_node, llm=llm)
synthesizer = partial(synthesizer_node, llm=llm)
critic = partial(critic_node, llm=llm)
reporter = partial(reporter_node, llm=llm)

def route_planning(state: GraphState):
    plan = state["research_plan"]
    if plan.step_type == "search":
        return "researcher"
    elif plan.step_type == "synthesize":
        return "synthesizer"

def route_critic(state: GraphState):
    if state["research_critic"].is_valid:
        return "reporter"
    else:
        return "synthesizer"

graph_builder = StateGraph(GraphState)

graph_builder.add_node("planner", planner)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("synthesizer", synthesizer)
graph_builder.add_node("critic", critic)
graph_builder.add_node("reporter", reporter)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("researcher", "planner") 
graph_builder.add_edge("synthesizer", "critic")

graph_builder.add_conditional_edges(
    "planner",
    route_planning,
    {
        "researcher": "researcher",
        "synthesizer": "synthesizer"
    }
)

graph_builder.add_conditional_edges(
    "critic",
    route_critic,
    {
        "synthesizer": "synthesizer",
        "reporter": "reporter"
    }
)

graph_builder.set_finish_point("reporter")

app = graph_builder.compile()

if __name__ == "__main__":
    initial_topic = "Using AI to mitigate urban heat island effect"

    inputs = {"initial_topic": initial_topic}

    final_state = None
    final_proposal = None

    for step in app.stream(inputs, {"recursion_limit": 15}):
        node_name, output = next(iter(step.items()))
        print(f"\n--- Executed Node: {node_name} ---")
        final_state = step
    
    if final_state:
        last_output = next(iter(final_state.values()))
        final_proposal = last_output.get("final_proposal")
    else:
        final_proposal = None

    if final_proposal:
        print("\n\n--- FINAL RESEARCH PROPOSAL ---")
        print(f"Title: {final_proposal.Title}")
        print("\nIntroduction:")
        print(final_proposal.Introduction)
        print("\nProblem Statement:")
        print(final_proposal.ResearchProblem)
        print("\nMethodology:")
        print(final_proposal.Methodology)
        print("\nExpected Outcomes:")
        print(final_proposal.ExpectedOutcomes)
        print("\nReferences:")
        for ref in final_proposal.References:
            print(f"- {ref['title']}: {ref['uri']}")
    else:
        print("\n\n--- PROCESS FINISHED WITHOUT A FINAL PROPOSAL ---")