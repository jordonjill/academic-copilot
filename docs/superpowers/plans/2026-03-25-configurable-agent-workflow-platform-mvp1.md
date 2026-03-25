# Configurable Agent/Workflow Platform (MVP-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven supervisor/subagent/workflow runtime that makes `proposal_v2` the default proposal path, supports workflow confirmation interruption fallback to dynamic mode, enforces same-agent retry caps, and ships SQLite-backed memory compression/history in MVP-1.

**Architecture:** Introduce a runtime layer (`config_registry`, `agent_runtime`, `workflow_runtime`, `orchestrator`) that loads YAML agent/workflow specs and executes them with guarded hooks and tool IDs. Keep existing FastAPI surface while extending `/chat` and adding `/admin/reload`. Replace hardcoded proposal routing with `proposal_v2` config while preserving rollback compatibility.

**Tech Stack:** Python 3.12, FastAPI, LangChain/LangGraph, Pydantic v2, SQLite, PyYAML, pytest

---

## Scope Check

This plan implements only the approved sub-project 1 (configurable agent/workflow platform) with memory included in MVP-1. Deferred epics (`skills/tools/mcp unification depth`, governance expansion) remain in backlog and are not implemented here.

## File Structure Map

### New Files

- `backend/config/agents/planner_proposal.yaml`: planner subagent spec for `proposal_v2`
- `backend/config/agents/researcher_proposal.yaml`: researcher subagent spec
- `backend/config/agents/synthesizer_proposal.yaml`: synthesizer subagent spec
- `backend/config/agents/critic_proposal.yaml`: critic subagent spec
- `backend/config/agents/reporter_proposal.yaml`: reporter subagent spec
- `backend/config/workflows/proposal_v2.yaml`: full workflow graph config with loop/branch limits
- `backend/src/application/runtime/spec_models.py`: Pydantic schemas for AgentSpec/WorkflowSpec
- `backend/src/application/runtime/config_registry.py`: YAML loader, validator, in-memory versioned registry
- `backend/src/application/runtime/hook_registry.py`: whitelist hook registry
- `backend/src/application/runtime/agent_runtime.py`: instantiate chain/react agents from AgentSpec
- `backend/src/application/runtime/workflow_runtime.py`: compile workflow spec into executable runtime graph
- `backend/src/application/runtime/orchestrator.py`: supervisor orchestration policy (workflow suggestion, interruption, dynamic mode)
- `backend/src/interfaces/api/routes/admin.py`: `POST /admin/reload`
- `backend/src/infrastructure/tools/docx_export.py`: export tool for Word output
- `backend/src/infrastructure/tools/pdf_export.py`: export tool for PDF output
- `backend/tests/conftest.py`: shared fixtures and test helpers
- `backend/tests/runtime/test_spec_models.py`: DSL validation tests
- `backend/tests/runtime/test_config_registry.py`: loader/reload behavior tests
- `backend/tests/runtime/test_workflow_runtime.py`: route/loop/timeout behavior tests
- `backend/tests/runtime/test_orchestrator_confirmation.py`: confirmation, interruption, retry-cap tests
- `backend/tests/memory/test_memory_pipeline.py`: raw/working/compression behavior tests
- `backend/tests/api/test_chat_and_admin_routes.py`: API behavior tests
- `backend/tests/tools/test_export_tools.py`: Word/PDF tool tests

### Modified Files

- `backend/src/interfaces/api/schemas.py`: add optional `workflow_id` and suggestion response data compatibility
- `backend/src/interfaces/api/routes/chat.py`: support workflow suggestion responses and explicit workflow routing
- `backend/src/interfaces/api/service.py`: integrate orchestrator/runtime registry and memory pipeline
- `backend/src/interfaces/api/routes/__init__.py`: register `admin` router
- `backend/main.py`: include `admin` router and ensure startup registry init
- `backend/src/domain/state/schema.py`: add confirmation/dynamic-mode/retry-tracking fields
- `backend/src/infrastructure/memory/sqlite_store.py`: add tables/columns for `raw_messages`, `working_context`, `compression_events`
- `backend/src/infrastructure/memory/stm.py`: refactor to keep raw history + compressed working context split
- `backend/src/infrastructure/tools/registry.py`: register `docx_export` and `pdf_export` tool IDs for binding
- `backend/src/infrastructure/tools/__init__.py`: export new tools
- `backend/src/application/graph.py`: make `proposal_v2` runtime-driven default proposal path, keep rollback toggle

### Optional Docs Update (same PR)

- `backend/README.md`: add config directories, reload endpoint, and new chat behavior notes

---

### Task 1: Create Runtime Spec Schemas (AgentSpec/WorkflowSpec)

**Files:**
- Create: `backend/src/application/runtime/spec_models.py`
- Test: `backend/tests/runtime/test_spec_models.py`

- [ ] **Step 1: Write the failing tests (@superpowers:test-driven-development)**

```python
# backend/tests/runtime/test_spec_models.py
import pytest
from pydantic import ValidationError
from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


def test_agent_spec_requires_core_fields():
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"id": "a1"})


def test_workflow_spec_requires_entry_and_nodes():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"id": "wf1", "name": "wf"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/runtime/test_spec_models.py -v`
Expected: FAIL with `ModuleNotFoundError` for runtime spec module.

- [ ] **Step 3: Write minimal schema implementation**

```python
# backend/src/application/runtime/spec_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0

class HooksConfig(BaseModel):
    pre_run: Optional[str] = None
    post_run: Optional[str] = None

class AgentSpec(BaseModel):
    id: str
    name: str
    mode: Literal["chain", "react"]
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    llm: LLMConfig
    hooks: Optional[HooksConfig] = None

class WorkflowSpec(BaseModel):
    id: str
    name: str
    entry_node: str
    nodes: Dict[str, dict]
    edges: List[dict]
    limits: Dict[str, int] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_spec_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/spec_models.py backend/tests/runtime/test_spec_models.py
git commit -m "feat(runtime): add agent/workflow spec models"
```

### Task 2: Implement Hook Whitelist Registry

**Files:**
- Create: `backend/src/application/runtime/hook_registry.py`
- Test: `backend/tests/runtime/test_config_registry.py`

- [ ] **Step 1: Write failing whitelist test**

```python
# add in backend/tests/runtime/test_config_registry.py
import pytest
from src.application.runtime.hook_registry import resolve_hook


def test_unknown_hook_rejected():
    with pytest.raises(KeyError):
        resolve_hook("not_registered_hook")
```

- [ ] **Step 2: Run test to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py::test_unknown_hook_rejected -v`
Expected: FAIL with missing module/function.

- [ ] **Step 3: Implement whitelist resolver**

```python
# backend/src/application/runtime/hook_registry.py
from typing import Callable, Dict

HOOK_REGISTRY: Dict[str, Callable] = {}


def register_hook(name: str, fn: Callable) -> None:
    HOOK_REGISTRY[name] = fn


def resolve_hook(name: str) -> Callable:
    if name not in HOOK_REGISTRY:
        raise KeyError(f"Hook not registered: {name}")
    return HOOK_REGISTRY[name]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py::test_unknown_hook_rejected -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/hook_registry.py backend/tests/runtime/test_config_registry.py
git commit -m "feat(runtime): add whitelist hook registry"
```

### Task 3: Build Config Registry + Reload Semantics

**Files:**
- Create: `backend/src/application/runtime/config_registry.py`
- Create: `backend/config/agents/*.yaml`
- Create: `backend/config/workflows/proposal_v2.yaml`
- Test: `backend/tests/runtime/test_config_registry.py`

- [ ] **Step 1: Write failing load/reload tests**

```python
# backend/tests/runtime/test_config_registry.py
from src.application.runtime.config_registry import ConfigRegistry


def test_registry_loads_workflow_and_agents(tmp_path):
    reg = ConfigRegistry(config_root=tmp_path)
    report = reg.reload()
    assert "config_version" in report


def test_registry_partial_failure_isolated(tmp_path):
    reg = ConfigRegistry(config_root=tmp_path)
    report = reg.reload()
    assert "failed_objects" in report
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py -v`
Expected: FAIL with missing `ConfigRegistry`.

- [ ] **Step 3: Implement registry + baseline YAMLs**

```python
# backend/src/application/runtime/config_registry.py
class ConfigRegistry:
    def __init__(self, config_root):
        self.config_root = config_root
        self.version = 0
        self.agents = {}
        self.workflows = {}

    def reload(self):
        # load yaml -> validate -> isolate failures
        self.version += 1
        return {
            "config_version": self.version,
            "loaded_agents": list(self.agents.keys()),
            "loaded_workflows": list(self.workflows.keys()),
            "failed_objects": [],
        }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/config_registry.py backend/config/agents backend/config/workflows backend/tests/runtime/test_config_registry.py
git commit -m "feat(runtime): add config registry and proposal_v2 yaml specs"
```

### Task 4: Add Export Tools (Word/PDF) and Tool Registry Bindings

**Files:**
- Create: `backend/src/infrastructure/tools/docx_export.py`
- Create: `backend/src/infrastructure/tools/pdf_export.py`
- Modify: `backend/src/infrastructure/tools/registry.py`
- Modify: `backend/src/infrastructure/tools/__init__.py`
- Modify: `backend/pyproject.toml`
- Test: `backend/tests/tools/test_export_tools.py`

- [ ] **Step 1: Write failing export tool tests**

```python
# backend/tests/tools/test_export_tools.py
from pathlib import Path
from src.infrastructure.tools.docx_export import export_docx
from src.infrastructure.tools.pdf_export import export_pdf


def test_export_docx_creates_file(tmp_path):
    out = export_docx.invoke({"title": "T", "content": "hello", "output_path": str(tmp_path / "a.docx")})
    assert Path(out["path"]).exists()


def test_export_pdf_creates_file(tmp_path):
    out = export_pdf.invoke({"title": "T", "content": "hello", "output_path": str(tmp_path / "a.pdf")})
    assert Path(out["path"]).exists()
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/tools/test_export_tools.py -v`
Expected: FAIL with missing export tool modules.

- [ ] **Step 3: Implement minimal export tools and register IDs**

```python
# backend/src/infrastructure/tools/docx_export.py
from langchain_core.tools import tool

@tool
def export_docx(title: str, content: str, output_path: str) -> dict:
    from docx import Document
    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_paragraph(content)
    doc.save(output_path)
    return {"path": output_path}
```

```python
# backend/src/infrastructure/tools/pdf_export.py
from langchain_core.tools import tool

@tool
def export_pdf(title: str, content: str, output_path: str) -> dict:
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(output_path)
    c.drawString(72, 800, title)
    c.drawString(72, 780, content[:4000])
    c.save()
    return {"path": output_path}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/tools/test_export_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/infrastructure/tools/docx_export.py backend/src/infrastructure/tools/pdf_export.py backend/src/infrastructure/tools/registry.py backend/src/infrastructure/tools/__init__.py backend/pyproject.toml backend/tests/tools/test_export_tools.py
git commit -m "feat(tools): add docx/pdf export tools and registry bindings"
```

### Task 5: Build Agent Runtime from AgentSpec

**Files:**
- Create: `backend/src/application/runtime/agent_runtime.py`
- Test: `backend/tests/runtime/test_workflow_runtime.py`

- [ ] **Step 1: Write failing agent runtime tests**

```python
# backend/tests/runtime/test_workflow_runtime.py
from src.application.runtime.agent_runtime import build_agent_from_spec


def test_build_chain_agent_from_spec(chain_spec, fake_llm):
    agent = build_agent_from_spec(chain_spec, fake_llm, tool_resolver=lambda _: [])
    assert agent is not None
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py::test_build_chain_agent_from_spec -v`
Expected: FAIL with missing function.

- [ ] **Step 3: Implement agent runtime factory**

```python
# backend/src/application/runtime/agent_runtime.py
from src.application.agents.factory import create_subagent, AgentMode

def build_agent_from_spec(spec, llm, tool_resolver):
    tools = [tool_resolver(tid) for tid in spec.tools]
    tools = [t for t in tools if t is not None]
    mode = AgentMode.CHAIN if spec.mode == "chain" else AgentMode.REACT
    return create_subagent(mode, llm, prompt=spec.system_prompt, tools=tools, name=spec.id)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py::test_build_chain_agent_from_spec -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/agent_runtime.py backend/tests/runtime/test_workflow_runtime.py
git commit -m "feat(runtime): build agents from AgentSpec"
```

### Task 6: Build Workflow Runtime (Branch + Loop + Limits)

**Files:**
- Create: `backend/src/application/runtime/workflow_runtime.py`
- Test: `backend/tests/runtime/test_workflow_runtime.py`

- [ ] **Step 1: Write failing workflow routing/loop tests**

```python
# add in backend/tests/runtime/test_workflow_runtime.py

def test_workflow_routes_planner_to_researcher(workflow_runtime, proposal_state):
    nxt = workflow_runtime.next_node("planner", proposal_state)
    assert nxt in {"researcher", "synthesizer"}


def test_workflow_stops_on_loop_limit(workflow_runtime, proposal_state):
    proposal_state["_loop_count"] = 99
    with pytest.raises(RuntimeError):
        workflow_runtime.enforce_limits(proposal_state)
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py -v`
Expected: FAIL with missing runtime methods.

- [ ] **Step 3: Implement workflow compiler/runtime with limits**

```python
# backend/src/application/runtime/workflow_runtime.py
class WorkflowRuntime:
    def __init__(self, spec, agent_runner):
        self.spec = spec
        self.agent_runner = agent_runner

    def next_node(self, current_node, state):
        outgoing = [e for e in self.spec.edges if e["from"] == current_node]
        for edge in outgoing:
            cond = edge.get("condition")
            if cond is None:
                return edge["to"]
            field = cond.get("field")
            equals = cond.get("equals")
            if state.get(field) == equals:
                return edge["to"]
        raise RuntimeError(f"No valid transition from node: {current_node}")

    def enforce_limits(self, state):
        if state.get("_step_count", 0) > self.spec.limits.get("max_steps", 30):
            raise RuntimeError("max_steps exceeded")
        if state.get("_loop_count", 0) > self.spec.limits.get("max_loops", 6):
            raise RuntimeError("max_loops exceeded")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/workflow_runtime.py backend/tests/runtime/test_workflow_runtime.py
git commit -m "feat(runtime): add workflow runtime with branch/loop limits"
```

### Task 7: Implement Orchestrator Policy (Suggestion, Interruption, Dynamic Retry Cap)

**Files:**
- Create: `backend/src/application/runtime/orchestrator.py`
- Modify: `backend/src/domain/state/schema.py`
- Test: `backend/tests/runtime/test_orchestrator_confirmation.py`

- [ ] **Step 1: Write failing policy tests**

```python
# backend/tests/runtime/test_orchestrator_confirmation.py

def test_pending_confirmation_interrupted_by_new_question_enters_dynamic(orchestrator, state):
    state["pending_workflow_confirmation"] = True
    out = orchestrator.handle_user_input(state, "顺便帮我解释一下这个术语")
    assert out["orchestration_mode"] == "dynamic"
    assert out["pending_workflow_confirmation"] is False


def test_retry_cap_blocks_same_agent_loop(orchestrator, state):
    state["last_selected_agent_id"] = "researcher_proposal"
    state["agent_retry_counters"] = {"researcher_proposal": 2}
    out = orchestrator.select_next_agent(state)
    assert out != "researcher_proposal"
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_orchestrator_confirmation.py -v`
Expected: FAIL with missing orchestrator.

- [ ] **Step 3: Implement orchestrator policy and state fields**

```python
# backend/src/application/runtime/orchestrator.py
MAX_RETRIES_PER_AGENT = 2
CONFIRM_ACCEPT = {"yes", "use workflow", "使用"}
CONFIRM_REJECT = {"no", "don't use", "不使用"}

def should_discard_confirmation(user_text: str) -> bool:
    normalized = user_text.strip().lower()
    if normalized in CONFIRM_ACCEPT or normalized in CONFIRM_REJECT:
        return False
    return True

class SupervisorOrchestrator:
    def handle_user_input(self, state, text):
        if state.get("pending_workflow_confirmation") and should_discard_confirmation(text):
            state["pending_workflow_confirmation"] = False
            state["suggested_workflow_id"] = None
            state["orchestration_mode"] = "dynamic"
        return state
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_orchestrator_confirmation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/runtime/orchestrator.py backend/src/domain/state/schema.py backend/tests/runtime/test_orchestrator_confirmation.py
git commit -m "feat(orchestrator): add confirmation interruption and per-agent retry cap"
```

### Task 8: Implement MVP-1 Memory Pipeline with SQLite (Raw + Working + Compression Events)

**Files:**
- Modify: `backend/src/infrastructure/memory/sqlite_store.py`
- Modify: `backend/src/infrastructure/memory/stm.py`
- Test: `backend/tests/memory/test_memory_pipeline.py`

- [ ] **Step 1: Write failing memory persistence/compression tests**

```python
# backend/tests/memory/test_memory_pipeline.py

def test_raw_messages_are_always_persisted(sqlite_store, long_state, fake_llm):
    out = stm_compression_node(long_state, fake_llm)
    rows = sqlite_store.get_session_messages(long_state["session_id"], backbone_only=False)
    assert len(rows) >= 2


def test_compression_rewrites_working_context_but_keeps_history(sqlite_store, long_state, fake_llm):
    out = stm_compression_node(long_state, fake_llm)
    assert out["stm_compressed"] is True
    assert any("Compressed Context" in str(m.content) for m in out["messages"])
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/memory/test_memory_pipeline.py -v`
Expected: FAIL for missing schema fields/tables.

- [ ] **Step 3: Update SQLite schema + memory node behavior**

```python
# sqlite schema additions (example)
CREATE TABLE IF NOT EXISTS compression_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    pre_tokens INTEGER NOT NULL,
    post_tokens INTEGER NOT NULL,
    summary_hash TEXT,
    created_at TEXT NOT NULL
);
# persist raw_messages each turn
# persist working_context snapshots after compression decision
```

```python
# stm.py behavior
# 1) persist raw
# 2) token check on working_context
# 3) if needed: summarize + recent_k -> new working_context
# 4) keep raw history untouched
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/memory/test_memory_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/infrastructure/memory/sqlite_store.py backend/src/infrastructure/memory/stm.py backend/tests/memory/test_memory_pipeline.py
git commit -m "feat(memory): implement sqlite raw/working/compression pipeline for mvp"
```

### Task 9: API Integration (`/chat` + `/admin/reload`) and Service Wiring

**Files:**
- Create: `backend/src/interfaces/api/routes/admin.py`
- Modify: `backend/src/interfaces/api/routes/__init__.py`
- Modify: `backend/src/interfaces/api/schemas.py`
- Modify: `backend/src/interfaces/api/routes/chat.py`
- Modify: `backend/src/interfaces/api/service.py`
- Modify: `backend/main.py`
- Test: `backend/tests/api/test_chat_and_admin_routes.py`

- [ ] **Step 1: Write failing API tests**

```python
# backend/tests/api/test_chat_and_admin_routes.py

def test_admin_reload_returns_report(client):
    resp = client.post("/admin/reload", headers={"Authorization": "Bearer test"})
    assert resp.status_code == 200
    assert "config_version" in resp.json()["data"]


def test_chat_accepts_workflow_id(client):
    resp = client.post("/chat", json={"message": "x", "workflow_id": "proposal_v2", "model_type": "ollama"}, headers={"Authorization": "Bearer test"})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd backend && uv run pytest tests/api/test_chat_and_admin_routes.py -v`
Expected: FAIL (`/admin/reload` not found, schema missing workflow_id).

- [ ] **Step 3: Implement routes and service integration**

```python
# schemas.py
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    session_id: Optional[str] = None
    model_type: str = "ollama"
    workflow_id: Optional[str] = None
```

```python
# admin.py
@router.post("/admin/reload")
async def reload_configs(_: str = Depends(verify_access_key)):
    report = config_registry.reload()
    return {"success": True, "data": report}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/api/test_chat_and_admin_routes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/interfaces/api/routes/admin.py backend/src/interfaces/api/routes/__init__.py backend/src/interfaces/api/schemas.py backend/src/interfaces/api/routes/chat.py backend/src/interfaces/api/service.py backend/main.py backend/tests/api/test_chat_and_admin_routes.py
git commit -m "feat(api): add admin reload and workflow-aware chat schema/service"
```

### Task 10: Replace Legacy Proposal Route with `proposal_v2` and Add Rollback Switch

**Files:**
- Modify: `backend/src/application/graph.py`
- Modify: `backend/src/interfaces/api/service.py`
- Test: `backend/tests/runtime/test_workflow_runtime.py`

- [ ] **Step 1: Write failing default-route test**

```python
# add in backend/tests/runtime/test_workflow_runtime.py

def test_default_proposal_route_uses_proposal_v2(orchestrator, state):
    state["current_intent"] = type("I", (), {"intent": "PROPOSAL_GEN"})()
    route = orchestrator.resolve_route(state)
    assert route == "proposal_v2"
```

- [ ] **Step 2: Run test to verify fail**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py::test_default_proposal_route_uses_proposal_v2 -v`
Expected: FAIL with current legacy route behavior.

- [ ] **Step 3: Implement replacement + rollback guard**

```python
# graph/service behavior
# default proposal intent -> proposal_v2 runtime
# env flag `PROPOSAL_V2_ROLLBACK=1` -> route to legacy proposal_workflow
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py::test_default_proposal_route_uses_proposal_v2 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/application/graph.py backend/src/interfaces/api/service.py backend/tests/runtime/test_workflow_runtime.py
git commit -m "feat(workflow): make proposal_v2 default with rollback switch"
```

### Task 11: Final Integration Verification and Documentation

**Files:**
- Modify: `backend/README.md`
- Optional modify: `README.md`

- [ ] **Step 1: Write/adjust docs checklist test (manual assertion file)**

```text
- /admin/reload documented
- /chat workflow_id documented
- confirmation interruption behavior documented
- proposal_v2 default route documented
- memory pipeline behavior documented
```

- [ ] **Step 2: Run full test suite**

Run: `cd backend && uv run pytest -v`
Expected: PASS, no failing tests.

- [ ] **Step 3: Smoke-run server startup**

Run: `cd backend && uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload`
Expected: startup log includes config registry init, routes mounted.

- [ ] **Step 4: Update docs with exact behavior**

```markdown
# backend/README.md additions
- `POST /admin/reload`
- `POST /chat` optional `workflow_id`
- workflow suggestion interruption fallback
- dynamic mode retry cap
```

- [ ] **Step 5: Commit**

```bash
git add backend/README.md README.md
git commit -m "docs: document mvp1 configurable runtime and api behavior"
```

## Execution Notes

- Keep each task in a separate commit.
- Do not combine unrelated refactors.
- Prefer rollback-safe, incremental merges.
- If an implementation detail conflicts with spec, update spec first, then continue.

## Suggested Execution Command Sequence

1. `cd backend && uv run pytest tests/runtime/test_spec_models.py -v`
2. `cd backend && uv run pytest tests/runtime/test_config_registry.py -v`
3. `cd backend && uv run pytest tests/runtime/test_workflow_runtime.py -v`
4. `cd backend && uv run pytest tests/runtime/test_orchestrator_confirmation.py -v`
5. `cd backend && uv run pytest tests/memory/test_memory_pipeline.py -v`
6. `cd backend && uv run pytest tests/tools/test_export_tools.py -v`
7. `cd backend && uv run pytest tests/api/test_chat_and_admin_routes.py -v`
8. `cd backend && uv run pytest -v`
