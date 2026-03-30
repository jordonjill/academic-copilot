from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ---- Structured artifacts (kept for reusable output contracts) ----

class ResearchPlan(BaseModel):
    has_enough_content: bool
    step_type: Literal["search", "synthesize"]
    query: Optional[str] = None


class Resource(BaseModel):
    uri: str
    title: str
    content: str


class ResearchCreation(BaseModel):
    research_gap: str
    research_idea: str


class ResearchCritic(BaseModel):
    is_valid: bool
    feedback: Optional[str] = None


class FinalProposal(BaseModel):
    Title: str
    Introduction: str
    ResearchProblem: str
    Methodology: str
    ExpectedOutcomes: str
    References: List[Dict[str, str]]


class UserProfile(BaseModel):
    user_id: str = "default"
    research_domains: List[str] = Field(default_factory=list)
    preferred_methodologies: List[str] = Field(default_factory=list)
    known_tools: List[str] = Field(default_factory=list)
    past_topics: List[str] = Field(default_factory=list)
    custom_facts: List[str] = Field(default_factory=list)
    raw_memory_md: str = ""


class IntentClassification(BaseModel):
    intent: Literal["CHITCHAT", "PROPOSAL_GEN", "SURVEY_WRITE", "CLARIFY_NEEDED"]
    confidence: float = Field(ge=0.0, le=1.0)
    workflow_topic: Optional[str] = None
    clarification_question: Optional[str] = None


class RuntimeInput(BaseModel):
    user_text: str
    user_id: str = "default"
    session_id: str


class RuntimeContext(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    memory_summary: str = ""


class RuntimeMeta(BaseModel):
    mode: Literal["dynamic", "workflow"] = "dynamic"
    workflow_id: Optional[str] = None
    current_node: Optional[str] = None
    step_count: int = 0
    loop_count: int = 0
    status: Literal["idle", "running", "completed", "failed"] = "idle"


class RuntimeIO(BaseModel):
    last_model_output: Optional[str] = None
    last_tool_outputs: List[str] = Field(default_factory=list)


class RuntimeArtifacts(BaseModel):
    topic: Optional[str] = None
    shared: Dict[str, Any] = Field(default_factory=dict)


class RuntimeOutput(BaseModel):
    final_text: Optional[str] = None
    final_structured: Optional[Dict[str, Any]] = None


class RuntimeErrors(BaseModel):
    last_error: Optional[str] = None


class GlobalState(MessagesState):
    """
    Simplified runtime state contract.
    """

    input: RuntimeInput
    context: RuntimeContext
    runtime: RuntimeMeta
    io: RuntimeIO
    artifacts: RuntimeArtifacts
    output: RuntimeOutput
    errors: RuntimeErrors
