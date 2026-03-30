from __future__ import annotations

from langchain_core.prompts import PromptTemplate

# NOTE:
# Agent orchestration prompts (planner/researcher/critic/reporter/supervisor) are
# now managed in YAML under backend/config/agents/*.yaml.
# This module only keeps prompts used by the optional STM/LTM memory pipeline.


# =============================================================
# STM compression prompt
# =============================================================
STM_COMPRESSION_PROMPT_TEMPLATE = """
You are a conversation summarizer for an Academic Copilot system. Compress the following conversation history into a concise summary that preserves all critical academic context.

## Conversation to Compress
{conversation_to_compress}

## Compression Requirements
1. Preserve: research topics discussed, key findings, user preferences, decisions made
2. Preserve: any specific names, methods, tools, or datasets mentioned
3. Discard: raw web search results, repetitive content, tool call details
4. Format: Write as a flowing paragraph or structured bullet points
5. Language: Match the language of the conversation (Chinese or English)
6. Length: Maximum 500 tokens

## Output
Write ONLY the compressed summary. No preamble or explanation.
"""

STM_COMPRESSION_PROMPT = PromptTemplate.from_template(STM_COMPRESSION_PROMPT_TEMPLATE)


# =============================================================
# LTM fact extraction prompt
# =============================================================
LTM_EXTRACTION_PROMPT_TEMPLATE = """
You are a fact extractor for an Academic Copilot's long-term memory system. Extract persistent facts about the user from this conversation.

## Conversation Backbone
{conversation_backbone}

## Extraction Categories
Extract facts into these 6 categories. Only include facts that are EXPLICITLY stated or clearly implied. Leave a list empty if no facts found.

Output ONLY a valid JSON object with these exact keys:
{{
    "research_domains": [],        // Academic fields (e.g., "NLP", "Civil Engineering", "Quantum Computing")
    "methodologies": [],           // Research methods (e.g., "Finite Element Analysis", "RAG", "Transformer")
    "tools_and_frameworks": [],    // Specific tools (e.g., "LangChain", "PyTorch", "ABAQUS")
    "past_topics": [],             // Research topics explored (e.g., "Urban Heat Island Effect mitigation using AI")
    "writing_preferences": [],     // Writing style preferences (e.g., "prefers English abstracts", "uses IEEE format")
    "custom_facts": []             // Other persistent facts about the user
}}

## Conversation to Analyze
{conversation_backbone}

Output only the JSON object, no other text.
"""

LTM_EXTRACTION_PROMPT = PromptTemplate.from_template(LTM_EXTRACTION_PROMPT_TEMPLATE)
