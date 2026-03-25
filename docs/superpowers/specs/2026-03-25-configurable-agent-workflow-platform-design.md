# Configurable Agent/Workflow Platform Design (MVP-1)

- Date: 2026-03-25
- Project: Academic Copilot
- Scope: Sub-project 1 of platform modernization (configurable agent/workflow runtime)
- Status: Approved design draft (ready for implementation planning)

## 1. Background and Goal

The current backend uses a mostly hardcoded graph topology:

- Fixed supervisor routing
- Fixed built-in workflows (`proposal_workflow`, `survey_workflow`)
- Fixed agent/tool bindings in code

The goal of MVP-1 is to move to a configuration-driven orchestration model while preserving backward compatibility and existing behavior where possible.

## 2. In-Scope and Out-of-Scope

### 2.1 In-Scope (MVP-1)

- Single-user mode only (no multi-user isolation)
- Global shared config for agents/workflows
- Config-driven subagent definition (YAML)
- Config-driven workflow definition (YAML + controlled Python hooks)
- Support complex graph capabilities (branching/loops/advanced routing)
- Deliver one complex example workflow: `proposal_v2`
- Add manual config reload endpoint: `POST /admin/reload`
- Allow both workflow selection methods:
  - Explicit `workflow_id` from user request
  - Supervisor intent-driven suggestion with user confirmation
- If user rejects suggested workflow, fallback to supervisor dynamic subagent orchestration
- Memory behavior alignment:
  - Persist every Human/AI message
  - Compress only working context when threshold exceeded
  - Keep full historical conversation intact

### 2.2 Out-of-Scope (MVP-1)

- Multi-user tenancy and per-user config isolation
- User-authored arbitrary tool implementation code
- Arbitrary dynamic Python import from YAML
- UI for editing workflow/agent config
- Full skills-tools-mcp unification at protocol level

## 3. High-Level Architecture

### 3.1 New/Enhanced Components

1. Config Registry (new)
- Loads and validates YAML specs from:
  - `backend/config/agents/*.yaml`
  - `backend/config/workflows/*.yaml`
- Maintains in-memory indexes:
  - `agent_id -> AgentSpec`
  - `workflow_id -> WorkflowSpec`
- Exposes versioned runtime snapshot
- Supports manual reload via `POST /admin/reload`

2. Tool Catalog (enhanced)
- Unified tool-id catalog for runtime binding
- In MVP-1, workflows/agents bind by tool IDs only
- Expand built-in academic tools (implementation may be phased):
  - `web_search`
  - `arxiv_search`
  - `semantic_scholar_search`
  - `crossref_search`
  - `doi_resolver`
  - `citation_formatter`
  - `pdf_metadata_extract`

3. Agent Runtime (new)
- Creates runtime subagents from `AgentSpec`
- Supports:
  - `mode=chain`
  - `mode=react`
- Applies configured prompt, model parameters, tool bindings
- Executes optional pre/post hooks from whitelist registry

4. Workflow Runtime (new)
- Compiles `WorkflowSpec` into executable runtime graph
- Supports complex topology:
  - conditional branches
  - loops
  - router nodes
  - terminal nodes
- Enforces runtime safeguards:
  - max steps
  - max loops
  - timeout

5. Supervisor Orchestrator (refactor)
- Keeps intent classification as first stage
- Supports both selection modes:
  - explicit `workflow_id`
  - suggestion based on intent -> ask confirmation
- Handles rejection path:
  - switch to dynamic subagent orchestration mode

## 4. Configuration DSL

## 4.1 AgentSpec (`backend/config/agents/<id>.yaml`)

Required fields:

- `id`: unique agent id
- `name`: display name
- `mode`: `chain | react`
- `system_prompt`: system behavior prompt
- `tools`: list of tool IDs
- `llm`:
  - `provider`
  - `model`
  - `temperature`

Optional fields:

- `input_schema`
- `output_schema`
- `hooks`:
  - `pre_run`
  - `post_run`

Example:

```yaml
id: researcher_proposal
name: Proposal Researcher
mode: react
system_prompt: |
  You are a rigorous academic researcher specializing in systematic literature review.
tools:
  - arxiv_search
  - web_search
  - semantic_scholar_search
  - crossref_search
llm:
  provider: openai
  model: gpt-4o
  temperature: 0
hooks:
  pre_run: normalize_topic_hook
  post_run: dedup_resources_hook
```

## 4.2 WorkflowSpec (`backend/config/workflows/<id>.yaml`)

Required fields:

- `id`
- `name`
- `entry_node`
- `nodes`
- `edges`

Recommended control fields:

- `limits.max_steps`
- `limits.max_loops`
- `limits.timeout_ms`

Node types:

- `agent`: invoke subagent by `agent_id`
- `hook`: invoke whitelist hook by `hook_name`
- `router`: evaluate route key/condition
- `terminal`: mark completion output

Edge features:

- unconditional edge
- conditional edge (`condition`, `route_key`)
- loop edge with loop-limit checks

Example topology for `proposal_v2`:

- `planner -> researcher | synthesizer`
- `researcher -> planner`
- `synthesizer -> critic`
- `critic -> reporter | synthesizer`
- `reporter -> end`

## 4.3 Hook Registry (controlled)

- Introduce `src/application/hooks/registry.py`
- YAML can reference hook names only
- Runtime resolves via `HOOK_REGISTRY: Dict[str, Callable]`
- Unknown hook name causes object-level load failure
- Arbitrary module import from YAML is disabled

## 5. Runtime Orchestration Flow

## 5.1 Chat Entry Behavior

`POST /chat` gains optional `workflow_id`.

Priority order:

1. If `workflow_id` is explicitly provided and exists: run it directly
2. Else classify user intent in supervisor
3. If intent maps to available workflow (e.g., `proposal_v2`): ask user confirmation
4. On user acceptance: run selected workflow
5. On user rejection: switch to dynamic supervisor orchestration mode

Confirmation protocol (required for deterministic implementation):

- Workflow suggestion response must be structured as `type=workflow_suggestion` with:
  - `session_id`
  - `suggested_workflow_id`
  - `question`
  - `expires_in_turns` (default 2)
- Confirmation replies are interpreted only when `pending_workflow_confirmation=true`
- Accepted reply examples: `yes`, `use workflow`, `使用`
- Rejected reply examples: `no`, `don't use`, `不使用`
- If suggestion expires without confirmation, runtime falls back to dynamic mode

## 5.2 New Conversation State Fields

Add to global state:

- `pending_workflow_confirmation: bool`
- `suggested_workflow_id: Optional[str]`
- `orchestration_mode: Literal["workflow", "dynamic"]`
- `selected_subagents: List[str]` (execution trace)
- `confirmation_expires_at_turn: Optional[int]`

## 5.3 Dynamic Mode (No Explicit Workflow)

When user rejects suggested workflow:

- Supervisor decides next subagent at each turn
- Subagent selection depends on intent, current artifacts, and failure signals
- No static workflow graph is disclosed to user
- Execution trace is logged for observability and debugging
- Runtime guardrails apply:
  - `dynamic_max_substeps` (default 8)
  - `dynamic_idle_limit` (default 2 consecutive no-progress turns)
  - On guardrail breach: return structured clarification prompt to user

## 6. Memory Strategy (Aligned with Requirements)

## 6.1 Storage Layers

1. `raw_messages`
- Persist all Human/AI turns without loss

2. `working_context`
- Context window used for current model invocation
- Can be rewritten by compression

3. `compression_events`
- Audit log for each compression event:
  - trigger timestamp
  - source message range
  - summary hash/version
  - resulting context token count

## 6.2 End-of-Turn Procedure

At turn end:

1. Persist raw Human/AI messages
2. Estimate token count of current working context
3. If below threshold: continue unchanged
4. If above threshold:
  - summarize historical backbone
  - keep recent `k` turns raw
  - rebuild working context as `summary_block + recent_k`
5. Keep raw history intact regardless of compression

## 6.3 Long-Term Memory Extraction

Two extraction schedules:

- Session-end extraction (once per finished session)
- Daily extraction (periodic consolidation)

Persist extracted facts with provenance:

- `source_session_ids`
- `fact_type`
- `fact_content`
- `extracted_at`

## 7. Failure Handling and Guardrails

- Config parsing/validation failure:
  - disable only failed config object
  - keep healthy objects available
- Hook execution failure:
  - policy-based handling (`fail_fast` or `fallback_continue`)
- Dynamic mode cannot choose valid subagent:
  - fallback to chitchat and ask user for clarification
- Workflow runtime exceeds limit/timeout:
  - terminate with structured runtime error
- `POST /admin/reload` requires access control (existing access key mechanism)

## 8. API and Compatibility Changes

## 8.1 API

- Extend `POST /chat` request schema:
  - `workflow_id?: string`
- Add endpoint:
  - `POST /admin/reload`

`POST /admin/reload` response payload:

- `config_version`
- `loaded_agents`
- `loaded_workflows`
- `failed_objects`

`POST /chat` response extension for workflow suggestion:

- `type: "workflow_suggestion"`
- `message`: confirmation question
- `data.suggested_workflow_id`
- `data.expires_in_turns`

## 8.2 Backward Compatibility

- Existing proposal/survey/chat behavior remains available during migration
- Legacy hardcoded graph can coexist behind compatibility path
- `proposal_v2` is introduced incrementally and made default only after verification
- Existing clients that do not handle `type=workflow_suggestion` should still receive a human-readable `message`

## 9. Testing and Acceptance Criteria

## 9.1 Unit Tests

- YAML schema validation for AgentSpec and WorkflowSpec
- Hook whitelist resolution and rejection of unknown hook names
- Workflow compile tests (node/edge/routing integrity)
- Runtime loop/timeout guard tests
- Memory compression trigger and context reconstruction tests

## 9.2 Integration Tests

- `POST /admin/reload` success and partial-failure scenarios
- `POST /chat` with explicit `workflow_id`
- intent-based workflow suggestion -> user accepts path
- intent-based workflow suggestion -> user rejects -> dynamic mode path
- `proposal_v2` loop path (critic rejects then returns to synthesize)

## 9.3 Acceptance Criteria

- New agent/workflow can be enabled/disabled by config changes without code edits
- `proposal_v2` runs end-to-end under runtime limits
- User can reject suggested workflow and still complete task via dynamic orchestration
- Full raw conversation history remains intact after compression
- Reload endpoint reports deterministic object-level status

## 10. Rollout Plan (MVP-1)

Phase 1: Foundations

- Implement config schemas and registry
- Implement hook registry (whitelist-only)
- Extend tool catalog with ID-based binding

Phase 2: Runtime

- Implement agent runtime from AgentSpec
- Implement workflow runtime compiler/executor
- Add `proposal_v2` YAML example

Phase 3: Supervisor Integration

- Add workflow suggestion/confirmation state machine
- Add reject -> dynamic mode branch
- Extend `/chat` with explicit `workflow_id`

Phase 4: Memory alignment

- Separate `raw_messages` and `working_context`
- Add `compression_events`
- Add session-end and daily LTM extraction entrypoints

Phase 5: Stabilization

- Add integration tests
- Enable compatibility fallback
- Document operational runbook for reload and failure recovery

## 11. Backlog TODO (Post MVP-1)

The following items come from the original modernization vision but are intentionally deferred from this MVP-1 implementation scope. Keep them as tracked epics for subsequent spec/plan cycles.

### 11.1 Epic B: Unified Tooling Abstraction (skills/tools/mcp)

- Define whether `skills`, built-in `tools`, and `mcp` endpoints are represented as one logical tool type in runtime.
- Evaluate two architecture options:
  - unified abstraction layer with multiple adapters
  - standardized MCP-first contract for all tool providers
- Add capability metadata for each tool:
  - `tool_id`, `provider_type`, `permissions`, `cost_profile`, `latency_profile`
- Support user-defined tool registration flow (metadata + implementation binding) with security guardrails.
- Ensure subagent config can bind to any registered tool uniformly without changing agent runtime code.

### 11.2 Epic C: Memory System Re-architecture

- Persist both Human input and AI response for every turn in immutable raw history.
- Keep compression-only working context:
  - after each round, evaluate token usage
  - when threshold exceeded, inject compressed summary + recent turns into model context
  - retain all original historical turns for replay/audit
- Store compression artifacts:
  - compression trigger reason
  - pre/post token counts
  - summary version/hash
- Add long-term memory extraction schedules:
  - per-session end extraction
  - daily aggregation extraction
- Define merge and dedup policy between session-level and daily-level memory facts.

### 11.3 Epic D: Runtime Governance and Operations

- Add policy and permission model for dynamic subagent/workflow/tool execution.
- Add audit trail for:
  - supervisor decisions
  - selected subagents
  - called workflows
  - invoked tools
- Add observability:
  - per-node latency
  - per-tool error rate
  - token consumption by mode (`workflow` vs `dynamic`)
- Add config/version governance:
  - rollback to previous known-good config snapshot
  - change history and operator identity

## 12. Open Decisions Deferred to Implementation Plan

- Exact expression format for edge conditions (`route_key` vs expression DSL)
- First batch and order of new academic tools to implement concretely (MVP minimum: implement at least two adapters beyond existing `web_search` and `arxiv_search`; others may be staged)
- Whether `proposal_v2` immediately replaces old proposal workflow in default routing
- Detailed dynamic-mode supervisor policy (selection heuristics and retry policy)
