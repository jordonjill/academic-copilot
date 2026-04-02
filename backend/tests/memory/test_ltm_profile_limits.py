from __future__ import annotations

import json

from src.infrastructure.memory import ltm as ltm_module


def test_merge_profiles_applies_caps_and_normalization(monkeypatch):
    monkeypatch.setattr(ltm_module, "LTM_MAX_ITEMS_PER_CATEGORY", 2)
    monkeypatch.setattr(ltm_module, "LTM_PAST_TOPICS_MAX_ITEMS", 2)
    monkeypatch.setattr(ltm_module, "LTM_FACT_MAX_CHARS", 10)

    existing = {
        "research_domains": ["Natural Language Processing", "NLP"],
        "methodologies": ["RAG"],
        "tools_and_frameworks": ["PyTorch"],
        "past_topics": ["Topic-1", "Topic-2"],
        "writing_preferences": ["Use IEEE style"],
        "custom_facts": ["Prefers bilingual output"],
    }
    new_facts = {
        "research_domains": ["nlp", "Graph Neural Network"],
        "methodologies": ["RAG", "  Agentic   Planning  "],
        "tools_and_frameworks": ["  PyTorch ", "LangChain"],
        "past_topics": ["Topic-3", "Topic-4"],
        "writing_preferences": ["Use IEEE style", "Short paragraphs"],
        "custom_facts": ["Prefers bilingual output", "Needs citation links"],
    }

    merged = ltm_module._merge_profiles(existing, new_facts)
    assert merged["research_domains"] == ["NLP", "Graph Neur"]
    assert merged["methodologies"] == ["RAG", "Agentic Pl"]
    assert merged["tools_and_frameworks"] == ["PyTorch", "LangChain"]
    assert merged["past_topics"] == ["Topic-3", "Topic-4"]
    assert merged["writing_preferences"] == ["Use IEEE s", "Short para"]
    assert merged["custom_facts"] == ["Prefers bi", "Needs cita"]


def test_write_memory_md_enforces_total_char_cap(monkeypatch, tmp_path):
    users_root = (tmp_path / "users").resolve()
    monkeypatch.setattr(ltm_module, "_USERS_ROOT", users_root)
    monkeypatch.setattr(ltm_module, "LTM_MEMORY_MD_MAX_CHARS", 500)
    monkeypatch.setattr(ltm_module, "LTM_MAX_ITEMS_PER_CATEGORY", 50)
    monkeypatch.setattr(ltm_module, "LTM_PAST_TOPICS_MAX_ITEMS", 50)
    monkeypatch.setattr(ltm_module, "LTM_FACT_MAX_CHARS", 120)

    profile = {
        "research_domains": [f"Domain-{i}" for i in range(20)],
        "methodologies": [f"Method-{i}" for i in range(20)],
        "tools_and_frameworks": [f"Tool-{i}" for i in range(20)],
        "past_topics": [f"Topic-{i}" for i in range(20)],
        "writing_preferences": [f"Style-{i}" for i in range(20)],
        "custom_facts": [f"Fact-{i}" for i in range(20)],
    }
    content = ltm_module._write_memory_md("alice", profile)
    assert len(content) <= 500

    saved = (users_root / "alice" / "memory.md").read_text(encoding="utf-8")
    assert saved == content


def test_load_ltm_profile_for_supervisor_respects_prompt_cap(monkeypatch):
    monkeypatch.setattr(ltm_module, "MEMORY_PIPELINE_ENABLED", True)
    monkeypatch.setattr(ltm_module, "LTM_SUPERVISOR_PROFILE_MAX_CHARS", 120)

    fake_profile = {
        "research_domains": [f"domain-{i}" for i in range(8)],
        "methodologies": [f"method-{i}" for i in range(8)],
        "tools_and_frameworks": [f"tool-{i}" for i in range(8)],
        "past_topics": [f"topic-{i}" for i in range(8)],
        "writing_preferences": [f"style-{i}" for i in range(8)],
        "custom_facts": [f"fact-{i}" for i in range(8)],
    }
    monkeypatch.setattr(ltm_module, "_load_existing_profile", lambda user_id: fake_profile)

    payload = ltm_module.load_ltm_profile_for_supervisor("u1")
    assert payload
    assert len(payload) <= 120
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)


def test_load_ltm_profile_for_supervisor_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(ltm_module, "MEMORY_PIPELINE_ENABLED", False)
    assert ltm_module.load_ltm_profile_for_supervisor("u1") == ""


def test_merge_profiles_semantic_dedup_and_delta():
    existing = {
        "research_domains": [],
        "methodologies": [],
        "tools_and_frameworks": [],
        "past_topics": [],
        "writing_preferences": [],
        "custom_facts": [
            "requires experiments to include ablation studies, significance tests, error analysis, and failure case discussion",
        ],
    }
    new_facts = {
        "custom_facts": [
            "User requires experiments to include ablation studies, significance tests, error analysis and failure-case discussion.",
            "needs bilingual abstract",
        ]
    }

    merged, delta = ltm_module._merge_profiles_with_delta(existing, new_facts)
    assert len(merged["custom_facts"]) == 2
    assert any("bilingual abstract" in item for item in merged["custom_facts"])
    # Near-duplicate long sentence should not be added as a separate third item.
    assert len(delta["custom_facts"]) <= 2
