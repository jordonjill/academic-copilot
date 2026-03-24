from __future__ import annotations
from functools import partial
from typing import Dict

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END

from src.state import GlobalState
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.synthesizer import synthesizer_node
from src.agents.critic import critic_node
from src.agents.reporter import reporter_node


def route_planning(state: GlobalState) -> str:
    plan = state["research_plan"]
    if plan.step_type == "search":
        return "researcher"
    return "synthesizer"


def route_critic(state: GlobalState) -> str:
    if state["research_critic"].is_valid:
        return "reporter"
    return "synthesizer"


def build_proposal_subgraph(llm: BaseLanguageModel) -> StateGraph:
    """将现有 5-Agent Proposal 流程封装为 Sub-graph，输入/输出均为 GlobalState。"""
    planner = partial(planner_node, llm=llm)
    synthesizer = partial(synthesizer_node, llm=llm)
    critic = partial(critic_node, llm=llm)
    reporter = partial(reporter_node, llm=llm)

    builder = StateGraph(GlobalState)

    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher_node)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("critic", critic)
    builder.add_node("reporter", reporter)

    builder.set_entry_point("planner")
    builder.add_edge("researcher", "planner")
    builder.add_edge("synthesizer", "critic")

    builder.add_conditional_edges(
        "planner",
        route_planning,
        {"researcher": "researcher", "synthesizer": "synthesizer"},
    )
    builder.add_conditional_edges(
        "critic",
        route_critic,
        {"synthesizer": "synthesizer", "reporter": "reporter"},
    )

    builder.add_edge("reporter", END)

    return builder.compile()
