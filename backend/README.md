# Backend (Academic Copilot)

FastAPI backend for a config-driven supervisor/subagent/workflow runtime.

## Architecture

- `application/runtime/`: spec models, registry, runtime engine, workflow guardrail
- `infrastructure/`: tools, memory, config, persistence
- `interfaces/api/`: HTTP routes and service wiring

## Directory Layout

```text
backend/
├── main.py
├── config/
│   ├── system/
│   ├── agents/
│   ├── workflows/
│   └── tools.yaml
├── src/
│   ├── application/
│   │   └── runtime/
│   ├── infrastructure/
│   │   ├── config/
│   │   ├── memory/
│   │   └── tools/
│   └── interfaces/
│       └── api/
└── data/
```

## Schema Layering Convention

- Runtime state contract lives in `application/runtime/contracts/state_types.py` (framework-level, generic).
- API request/response schema lives in `interfaces/api/schemas.py` (transport boundary).
- Config schema for LLM/Agent/Workflow lives in `application/runtime/contracts/spec_models.py` (YAML contract).
- Workflow-specific or business-specific structured schema must live with the owning module/workflow, not in a global shared domain layer.

## Environment Variables

Copy `.env.example` to `.env` and set:

- `ACCESS_KEY` (for `/chat` and `/health`)
- `ADMIN_ACCESS_KEY` (for `/admin/*`)
- `OPENAI_API_KEY` (required if any configured LLM profile uses OpenAI-compatible auth)
- `DEEPSEEK_API_KEY` (optional, if using DeepSeek profile)
- `QWEN_API_KEY` (optional, if using Qwen profile)
- `OLLAMA_API_KEY` (optional, if using Ollama profile; can be any non-empty placeholder for local no-auth deployments)
- `TAVILY_API_KEY` (recommended for web search fallback; optional if you only rely on arXiv source)
- `JINA_API_KEY` (optional)
- `ZOTERO_API_KEY` (optional)
- Runtime/memory/tool envs shown in `.env.example` (`SUPERVISOR_MAX_*`, `WORKFLOW_MAX_*`, `CHAT_TURN_TIMEOUT_SECONDS`, `LLM_REQUEST_TIMEOUT_SECONDS`, `CHAT_MAX_WORKERS`, etc.)
- Rate-limit envs: `CHAT_RATE_LIMIT_ENABLED`, `CHAT_RATE_LIMIT_REQUESTS`, `CHAT_RATE_LIMIT_WINDOW_SECONDS`

## Run

From repository root:

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Static frontend serving behavior:

- If `../frontend/dist` exists:
  - `/` -> `frontend/dist/index.html`
  - `/assets/*` -> `frontend/dist/assets/*`
- Otherwise (legacy fallback):
  - `/` -> `frontend/index.html`
  - `/static/*` -> files under `frontend/`

## API Endpoints

- `POST /chat` (Bearer auth via `ACCESS_KEY`)
  - request: `message`, optional `workflow_id`, optional `session_id`, optional `user_id`
  - in-memory sliding-window rate limit can be enabled via `CHAT_RATE_LIMIT_*`
- `POST /chat/stream` (Bearer auth via `ACCESS_KEY`)
  - SSE stream endpoint for realtime status/step/final events
  - emits `connected`, `status`, `step`, `completion`, `error` (+ heartbeat comments)
- `GET /health` (Bearer auth via `ACCESS_KEY`)
- `POST /admin/reload` (Bearer auth via `ADMIN_ACCESS_KEY`)
  - reload tools + runtime config
- `POST /admin/reload-runtime` (Bearer auth via `ADMIN_ACCESS_KEY`)
- `POST /admin/reload-tools` (Bearer auth via `ADMIN_ACCESS_KEY`)

## Config & Validation

Runtime config is loaded from:

- `backend/config/system/*.yaml` (system-internal agents, e.g. `supervisor`)
- `backend/config/agents/*.yaml` (user-editable subagents)
- `backend/config/workflows/*.yaml`

Tools config is loaded from:

- `backend/config/tools.yaml`

Startup and runtime reload perform binding validation:

- Agent `tools` must reference enabled `tool_id` entries in `tools.yaml`
- Workflow `agent` nodes must reference existing `agent_id`
- Agent-bound LLM profiles must not contain unresolved `${ENV_VAR}` placeholders
- Agent-bound LLM `api_key_env` must point to a non-empty environment variable at reload/startup

Validation issues are returned in reload responses under `data.runtime.failed`.
At startup, validation issues stop the app from booting.

## Runtime Behavior

- Supervisor can:
  - answer directly
  - run subagents
  - start a workflow
- Explicit `workflow_id` in `/chat` runs that workflow directly.
- Workflow runtime enforces edge constraints and step/loop limits.

## Timeout & Concurrency Tuning

- `CHAT_TURN_TIMEOUT_SECONDS` is the API-layer timeout (`asyncio.wait_for`) for one `/chat` or `/chat/stream` turn.
- `SUPERVISOR_MAX_WALL_TIME_SECONDS` limits supervisor loop wall-clock time.
- `WORKFLOW_MAX_WALL_TIME_SECONDS` limits workflow loop wall-clock time.
- `LLM_REQUEST_TIMEOUT_SECONDS` limits each single model call (passed into LLM client).
- `CHAT_MAX_WORKERS` controls the default threadpool size for chat runtime execution.

Recommended relation:

- `LLM_REQUEST_TIMEOUT_SECONDS < CHAT_TURN_TIMEOUT_SECONDS`
- `CHAT_TURN_TIMEOUT_SECONDS < SUPERVISOR_MAX_WALL_TIME_SECONDS`
- `CHAT_TURN_TIMEOUT_SECONDS < WORKFLOW_MAX_WALL_TIME_SECONDS`

Operational note:

- Runtime now prefers async `ainvoke` path for supervisor/workflow agent execution, reducing dependence on threadpool workers for LLM calls.
- Tool budgets are enforced by runtime wrappers; when a budget is exhausted, tool calls return a budget-exceeded payload to the agent instead of executing the external call.

## Memory Pipeline

Memory is on the main chat path:

- STM (short-term):
  - Before each turn: load latest `working_context` snapshot from SQLite into runtime messages
  - After each turn: persist raw/backbone/context snapshots and apply compression when threshold is exceeded
  - Compression trigger is token-based (`STM_TOKEN_THRESHOLD`)
  - Compression target keeps:
    - summary budget (`STM_SUMMARY_TARGET_TOKENS`)
    - recent context budget (`STM_RECENT_TARGET_TOKENS`)
  - If token budgeting is unavailable, fallback to count-based recent keep (`STM_KEEP_RECENT`)
  - SQLite tables: `raw_messages`, `working_context`, `compression_events`
- LTM (long-term):
  - Triggered after STM compression events
  - Extracted facts are merged and written to `data/users/<user_id>/memory.md`
  - A compact memory summary is injected into supervisor context in later turns
  - Injection payload key: `ltm_profile`

Relevant memory/window envs:

- `STM_TOKEN_THRESHOLD`
- `STM_SUMMARY_TARGET_TOKENS`
- `STM_RECENT_TARGET_TOKENS`
- `STM_KEEP_RECENT` (fallback)
- `SUPERVISOR_MESSAGES_TOKEN_CAP`
- `SUBAGENT_MESSAGES_TOKEN_CAP`

Conversation persistence details:

- SQLite database defaults to `data/conversations.db` (configurable by `CONVERSATION_DB`).
- `messages` / `raw_messages` persist the conversation backbone (`HumanMessage` / `AIMessage` text).
- Subagent/workflow final returned text is appended into supervisor conversation messages and therefore persisted.
- Supervisor internal decision JSON is not persisted as a standalone DB field; it only affects orchestration and resulting messages/artifacts.
- Workflow internal per-node transient state is isolated during execution; persisted conversation stores the returned output text, while detailed runtime traces live in in-memory/artifact payloads for the turn.

## Troubleshooting

### 1) `504 Gateway Timeout` on workflow calls

Common cause: total workflow latency (LLM + tool calls + loops) exceeds `CHAT_TURN_TIMEOUT_SECONDS`.

Check and tune:

- Increase `CHAT_TURN_TIMEOUT_SECONDS` (API turn timeout).
- Keep `LLM_REQUEST_TIMEOUT_SECONDS` lower than `CHAT_TURN_TIMEOUT_SECONDS`.
- Reduce workflow complexity (`max_steps`, loop count, expensive nodes).
- Reduce tool budgets if a node tends to over-call tools.

Quick check:

```bash
curl -s http://127.0.0.1:8000/health -H "Authorization: Bearer $ACCESS_KEY"
```

### 2) `curl code=000` or `Operation timed out`

This means no HTTP response was received in time (server not reachable or request timed out on client side).

Check:

- Backend is running on the same `BASE_URL`/port used by script.
- `ACCESS_KEY` matches backend env.
- `CURL_MAX_TIME_SECONDS` in `scripts/api_e2e.sh` is large enough for workflow tests.

Example:

```bash
cd backend
ACCESS_KEY=123 CURL_MAX_TIME_SECONDS=600 ./scripts/api_e2e.sh
```

### 3) arXiv `429` / `503` / `source_timeout`

This is upstream instability/rate-limit from arXiv and is expected in bursts.

Current behavior:

- Tool logs warnings such as:
  - `scholar_search.arxiv_rate_limited`
  - `scholar_search.source_timeout`
- Runtime degrades to web search results when available (`include_web=True` and Tavily configured).

If this happens frequently:

- Ensure `TAVILY_API_KEY` is set and valid.
- Keep `include_web` enabled for `scholar_search`.
- Avoid overly broad/high-fanout search prompts in one node.

### 4) repeated `tool.budget_exceeded` warnings

Example log:

- `tool.budget_exceeded tool_id=paper_fetch used=6 limit=6`

Meaning:

- The agent attempted tool calls after budget was exhausted.
- External tool execution is blocked by wrapper at that point.
- Warnings can repeat because model may retry in the same ReAct loop before finishing.

This is not extra external API cost after the limit; it is model retry behavior.

### 5) No web-search results, only arXiv logs

Check:

- `TAVILY_API_KEY` is present in backend runtime env.
- Tool config keeps web search enabled for `scholar_search`.
- Look for startup/runtime logs indicating web search branch is active (for example `include_web=True`).
