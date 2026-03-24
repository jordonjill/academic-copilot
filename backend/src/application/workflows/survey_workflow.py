"""
综述撰写工作流（SurveyWorkflow Sub-graph）。

拓扑：
  survey_researcher → survey_writer → END

接口契约：
  输入: GlobalState（读 initial_topic / messages）
  输出: 更新 GlobalState 的 current_draft_sections + final_output
"""
from __future__ import annotations
from functools import partial
from typing import Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from src.domain.state import GlobalState, Resource
from src.infrastructure.tools.crawl_search import crawl_search
from src.infrastructure.tools.arxiv_search import search_arxiv


# ─── Researcher 节点 ─────────────────────────────────────────────────────────

def survey_researcher_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """收集综述所需文献，利用 ArXiv + Crawl Search 双渠道检索。"""
    topic = state.get("initial_topic") or ""
    if not topic:
        from langchain_core.messages import HumanMessage
        msgs = state.get("messages", [])
        topic = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")

    # ArXiv 搜索
    arxiv_results = search_arxiv.invoke({"query": topic, "max_results": 10})
    new_resources: List[Resource] = []
    for r in arxiv_results:
        if isinstance(r, dict) and "error" not in r:
            try:
                new_resources.append(Resource(
                    uri=r.get("uri", ""),
                    title=r.get("title", "Unknown"),
                    content=r.get("content", ""),
                ))
            except Exception:
                pass

    # Web 补充搜索
    web_results = crawl_search.invoke({"query": f"{topic} survey literature review"})
    for r in web_results:
        if isinstance(r, dict) and "error" not in r:
            try:
                new_resources.append(Resource(**r))
            except Exception:
                pass

    return {
        "collected_materials": new_resources,   # operator.add
        "retrieved_resources": new_resources,   # operator.add
        "initial_topic": topic,
        "search_count": state.get("search_count", 0) + 1,
    }


# ─── Writer 节点 ─────────────────────────────────────────────────────────────

_SURVEY_WRITER_PROMPT = """You are an expert academic writer. Write a comprehensive literature survey on the topic: {topic}.

Use the following collected papers as sources:
{resources}

Structure the survey with these sections:
1. Introduction & Motivation
2. Background & Taxonomy
3. Key Methods & Approaches
4. Comparative Analysis & Limitations
5. Open Challenges & Future Directions
6. Conclusion

For each section, write 200-400 words. Use numbered citations [1], [2], etc.
Output as JSON: {{"sections": [{{"title": "...", "content": "..."}}], "references": [{{"title": "...", "uri": "..."}}]}}
"""

from langchain_core.prompts import PromptTemplate

_SURVEY_PROMPT_OBJ = PromptTemplate.from_template(_SURVEY_WRITER_PROMPT)


def survey_writer_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """生成综述各章节，输出到 current_draft_sections + final_output。"""
    topic = state.get("initial_topic", "")
    resources = state.get("retrieved_resources", [])

    resources_text = "\n\n".join(
        f"[{i+1}] {r.title}\n  URI: {r.uri}\n  Abstract: {r.content[:300]}"
        for i, r in enumerate(resources)
    )

    from langchain_core.output_parsers import JsonOutputParser
    chain = _SURVEY_PROMPT_OBJ | llm | JsonOutputParser()

    try:
        result = chain.invoke({"topic": topic, "resources": resources_text})
        sections: List[Dict] = result.get("sections", [])
        references: List[Dict] = result.get("references", [
            {"title": r.title, "uri": r.uri} for r in resources
        ])
    except Exception as e:
        print(f"[SurveyWriter] JSON parse failed: {e}")
        sections = [{"title": "Survey Draft", "content": "Survey generation encountered an error."}]
        references = [{"title": r.title, "uri": r.uri} for r in resources]

    final_output = {
        "type": "survey",
        "topic": topic,
        "sections": sections,
        "references": references,
    }

    reply = f"综述《{topic}》已生成，共 {len(sections)} 个章节。"

    return {
        "current_draft_sections": sections,
        "final_output": final_output,
        "messages": [AIMessage(content=reply)],
        "workflow_status": "completed",
    }


# ─── Sub-graph 构建 ───────────────────────────────────────────────────────────

def build_survey_subgraph(llm: BaseLanguageModel):
    """将综述撰写流程封装为 Sub-graph，输入/输出均为 GlobalState。"""
    researcher = partial(survey_researcher_node, llm=llm)
    writer = partial(survey_writer_node, llm=llm)

    builder = StateGraph(GlobalState)
    builder.add_node("survey_researcher", researcher)
    builder.add_node("survey_writer", writer)

    builder.set_entry_point("survey_researcher")
    builder.add_edge("survey_researcher", "survey_writer")
    builder.add_edge("survey_writer", END)

    return builder.compile()
