from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class ResearchPlan(BaseModel):

    """Represents a complete, structured plan for the research agent."""
    has_enough_content: bool
    step_type: Literal["search", "synthesize"] = Field(..., description="The categorical type of action this step represents.")
    query: Optional[str] = Field(default=None, description="The specific search query to be used, if applicable.")

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
    feedback: Optional[str] = Field(default=None, description="Feedback from the Critic agent if the idea fails validation.")

class FinalProposal(BaseModel):

    """Represents the final, structured research proposal document."""

    Title: str
    Introduction: str
    ResearchProblem: str
    Methodology: str
    ExpectedOutcomes: str
    References: List[Dict[str, str]]

class GraphState(MessagesState):

    """The central state for the automated research proposal generation system."""

    initial_topic: str
    retrieved_resources: List[Resource]
    research_plan: Optional[ResearchPlan]
    research_creation: Optional[ResearchCreation]
    research_critic: Optional[ResearchCritic]

    idea_validation_attempts: int
    search_count: int

    final_proposal: Optional[FinalProposal] = None