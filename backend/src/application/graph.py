"""
Academic Copilot 主图（Supervisor 路由层）。

拓扑：
  supervisor
      ├─ chitchat        → stm_compression → END
      ├─ proposal_workflow → stm_compression → END
      └─ survey_workflow   → stm_compression → END
"""
from __future__ import annotations
import os
from functools import partial

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END

from src.domain.state import GlobalState
from src.application.agents.supervisor import supervisor_node, route_by_intent
from src.application.agents.chitchat import chitchat_node
from src.application.workflows.proposal_workflow import build_proposal_subgraph
from src.application.workflows.survey_workflow import build_survey_subgraph
from src.infrastructure.memory.stm import stm_compression_node

load_dotenv()

LEGACY_PROPOSAL_ROUTE = "proposal_workflow"
PROPOSAL_V2_ROUTE = "proposal_v2"
PROPOSAL_V2_ROLLBACK_ENV = "PROPOSAL_V2_ROLLBACK"


def _is_proposal_v2_rollback_enabled() -> bool:
    return os.getenv(PROPOSAL_V2_ROLLBACK_ENV) == "1"


def resolve_supervisor_route(state: GlobalState) -> str:
    route = route_by_intent(state)
    if route != LEGACY_PROPOSAL_ROUTE:
        return route
    if _is_proposal_v2_rollback_enabled():
        return LEGACY_PROPOSAL_ROUTE
    return PROPOSAL_V2_ROUTE


def build_main_graph(llm):
    """构建并编译 Supervisor 主图。"""
    # --- Sub-graphs ---
    proposal_subgraph = build_proposal_subgraph(llm)
    survey_subgraph = build_survey_subgraph(llm)

    # --- 绑定 LLM 的节点 ---
    supervisor = partial(supervisor_node, llm=llm)
    chitchat = partial(chitchat_node, llm=llm)
    stm_compress = partial(stm_compression_node, llm=llm)

    builder = StateGraph(GlobalState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("chitchat", chitchat)
    builder.add_node(LEGACY_PROPOSAL_ROUTE, proposal_subgraph)
    builder.add_node(PROPOSAL_V2_ROUTE, proposal_subgraph)
    builder.add_node("survey_workflow", survey_subgraph)
    builder.add_node("stm_compression", stm_compress)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        resolve_supervisor_route,
        {
            "chitchat": "chitchat",
            LEGACY_PROPOSAL_ROUTE: LEGACY_PROPOSAL_ROUTE,
            PROPOSAL_V2_ROUTE: PROPOSAL_V2_ROUTE,
            "survey_workflow": "survey_workflow",
        },
    )

    # 所有路径收敛到 stm_compression
    for node in ["chitchat", LEGACY_PROPOSAL_ROUTE, PROPOSAL_V2_ROUTE, "survey_workflow"]:
        builder.add_edge(node, "stm_compression")

    builder.add_edge("stm_compression", END)

    return builder.compile()


# ── 默认实例（向后兼容 src/graph.py 被直接导入的场景）──
_llm = ChatOllama(model="llama3.1:8b", temperature=0)
# _llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

app = build_main_graph(_llm)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    import uuid

    result = app.invoke({
        "messages": [HumanMessage(content="帮我写一个关于大模型在土木工程中的应用的研究提案")],
        "user_id": "default",
        "user_profile": None,
        "current_intent": None,
        "workflow_status": "idle",
        "collected_materials": [],
        "retrieved_resources": [],
        "current_draft_sections": None,
        "final_output": None,
        "initial_topic": None,
        "research_plan": None,
        "research_creation": None,
        "research_critic": None,
        "idea_validation_attempts": 0,
        "search_count": 0,
        "final_proposal": None,
        "session_id": str(uuid.uuid4()),
        "stm_token_count": 0,
        "stm_compressed": False,
        "ltm_extraction_done": False,
    }, {"recursion_limit": 25})

    print("Final proposal:", result.get("final_proposal"))
