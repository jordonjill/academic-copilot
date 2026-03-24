from __future__ import annotations
import operator
from typing import List, Dict, Optional, Literal, Annotated, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ===== 保留原有 Pydantic 模型（零修改）=====

class ResearchPlan(BaseModel):
    """Represents a complete, structured plan for the research agent."""
    has_enough_content: bool
    step_type: Literal["search", "synthesize"] = Field(
        ..., description="The categorical type of action this step represents."
    )
    query: Optional[str] = Field(
        default=None, description="The specific search query to be used, if applicable."
    )


class Resource(BaseModel):
    """Represents a single resource retrieved from the web."""
    uri: str = Field(..., description="The Uniform Resource Identifier for the resource.")
    title: str = Field(..., description="The title of the resource.")
    content: str = Field(..., description="The resource's core content, crucial for subsequent analysis.")


class ResearchCreation(BaseModel):
    """Represents a validated research gap identified from the literature."""
    research_gap: str = Field(..., description="A clear and concise description of the identified research gap.")
    research_idea: str = Field(..., description="A single proposed research idea to address this gap.")


class ResearchCritic(BaseModel):
    """Represents a novel idea generated to address a specific research gap."""
    is_valid: bool
    feedback: Optional[str] = Field(
        default=None, description="Feedback from the Critic agent if the idea fails validation."
    )


class FinalProposal(BaseModel):
    """Represents the final, structured research proposal document."""
    Title: str
    Introduction: str
    ResearchProblem: str
    Methodology: str
    ExpectedOutcomes: str
    References: List[Dict[str, str]]


# ===== 新增：用户档案 (LTM 载体) =====

class UserProfile(BaseModel):
    """用户长期记忆档案，从 data/users/{user_id}/memory.md 加载。"""
    user_id: str = "default"
    research_domains: List[str] = Field(default_factory=list)
    preferred_methodologies: List[str] = Field(default_factory=list)
    known_tools: List[str] = Field(default_factory=list)
    past_topics: List[str] = Field(default_factory=list)
    custom_facts: List[str] = Field(default_factory=list)
    raw_memory_md: str = ""


# ===== 新增：Supervisor 路由控制 =====

class IntentClassification(BaseModel):
    """Supervisor LLM 输出的意图分类结果。"""
    intent: Literal["CHITCHAT", "PROPOSAL_GEN", "SURVEY_WRITE", "CLARIFY_NEEDED"]
    confidence: float = Field(ge=0.0, le=1.0)
    workflow_topic: Optional[str] = Field(
        default=None, description="剥离助词后的核心研究主题，仅在工作流意图时填充。"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="当意图为 CLARIFY_NEEDED 时，向用户提出的澄清问题。"
    )


# ===== 全局状态机 =====

class GlobalState(MessagesState):
    """Academic Copilot 全局状态机，兼容原 GraphState 所有字段。"""

    # ---- [控制流] Supervisor 字段 ----
    user_id: str
    user_profile: Optional[UserProfile]
    current_intent: Optional[IntentClassification]
    workflow_status: Literal["idle", "running", "completed", "failed"]

    # ---- [追加式] 跨工作流文献累积（operator.add reducer）----
    collected_materials: Annotated[List[Resource], operator.add]

    # ---- [工作流产出] 覆盖语义 ----
    current_draft_sections: Optional[List[Dict[str, str]]]
    final_output: Optional[Dict[str, Any]]

    # ---- [Proposal 工作流专用字段] 复用原 GraphState 命名 ----
    initial_topic: Optional[str]
    retrieved_resources: Annotated[List[Resource], operator.add]
    research_plan: Optional[ResearchPlan]
    research_creation: Optional[ResearchCreation]
    research_critic: Optional[ResearchCritic]
    idea_validation_attempts: int
    search_count: int
    final_proposal: Optional[FinalProposal]

    # ---- [记忆管道控制] ----
    session_id: str
    stm_token_count: int
    stm_compressed: bool
    ltm_extraction_done: bool
