from .schema import (
    GlobalState,
    UserProfile,
    IntentClassification,
    ResearchPlan,
    Resource,
    ResearchCreation,
    ResearchCritic,
    FinalProposal,
)

# 向后兼容别名
GraphState = GlobalState

__all__ = [
    "GlobalState",
    "GraphState",
    "UserProfile",
    "IntentClassification",
    "ResearchPlan",
    "Resource",
    "ResearchCreation",
    "ResearchCritic",
    "FinalProposal",
]
