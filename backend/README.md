# Backend (Academic Copilot)

FastAPI backend for a config-driven supervisor/subagent/workflow runtime.

## Architecture

- `domain/`: runtime state schema
- `application/runtime/`: spec models, registry, runtime engine, workflow guardrail
- `infrastructure/`: tools, memory, config, persistence
- `interfaces/api/`: HTTP routes and service wiring

## Directory Layout

```text
backend/
├── main.py
├── config/
│   ├── agents/
│   ├── workflows/
│   └── tools.yaml
├── src/
│   ├── application/
│   │   └── runtime/
│   ├── domain/
│   │   └── state/
│   ├── infrastructure/
│   │   ├── config/
│   │   ├── memory/
│   │   └── tools/
│   └── interfaces/
│       └── api/
└── data/
```

## Environment Variables

Copy `.env.example` to `.env` and set:

- `ACCESS_KEY` (for `/chat` and `/health`)
- `ADMIN_ACCESS_KEY` (for `/admin/*`)
- `OPENAI_API_KEY` (required if any configured LLM profile uses OpenAI-compatible auth)
- `DEEPSEEK_API_KEY` (optional, if using DeepSeek profile)
- `QWEN_API_KEY` (optional, if using Qwen profile)
- `OLLAMA_API_KEY` (optional, if using Ollama profile; can be any non-empty placeholder for local no-auth deployments)
- `TAVILY_API_KEY`
- `JINA_API_KEY` (optional)
- `ZOTERO_API_KEY` (optional)
- Runtime/memory/tool envs shown in `.env.example` (`SUPERVISOR_MAX_*`, `WORKFLOW_MAX_*`, `CHAT_TURN_TIMEOUT_SECONDS`, `LLM_REQUEST_TIMEOUT_SECONDS`, `CHAT_MAX_WORKERS`, etc.)

## Run

From repository root:

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend serves frontend static files from `../frontend`:

- `/` -> `frontend/index.html`
- `/static/*` -> files under `frontend/`

## API Endpoints

- `POST /chat` (Bearer auth via `ACCESS_KEY`)
  - request: `message`, optional `workflow_id`, optional `session_id`, optional `user_id`
- `GET /health` (Bearer auth via `ACCESS_KEY`)
- `POST /admin/reload` (Bearer auth via `ADMIN_ACCESS_KEY`)
  - reload tools + runtime config
- `POST /admin/reload-runtime` (Bearer auth via `ADMIN_ACCESS_KEY`)
- `POST /admin/reload-tools` (Bearer auth via `ADMIN_ACCESS_KEY`)

## Config & Validation

Runtime config is loaded from:

- `backend/config/agents/*.yaml`
- `backend/config/workflows/*.yaml`

Tools config is loaded from:

- `backend/config/tools.yaml`

Startup and runtime reload perform binding validation:

- Agent `tools` must reference enabled `tool_id` entries in `tools.yaml`
- Workflow `agent` nodes must reference existing `agent_id`

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

- `CHAT_TURN_TIMEOUT_SECONDS` is the API-layer timeout (`asyncio.wait_for`) for one `/chat` request.
- `SUPERVISOR_MAX_WALL_TIME_SECONDS` limits supervisor loop wall-clock time.
- `WORKFLOW_MAX_WALL_TIME_SECONDS` limits workflow loop wall-clock time.
- `LLM_REQUEST_TIMEOUT_SECONDS` limits each single model call (passed into LLM client).
- `CHAT_MAX_WORKERS` controls the default threadpool size for chat runtime execution.

Recommended relation:

- `CHAT_TURN_TIMEOUT_SECONDS < SUPERVISOR_MAX_WALL_TIME_SECONDS`
- `CHAT_TURN_TIMEOUT_SECONDS < WORKFLOW_MAX_WALL_TIME_SECONDS`

Operational note:

- `asyncio.wait_for` timeout returns control to API caller, but cannot forcibly terminate a running executor thread instantly. Keep `CHAT_MAX_WORKERS` conservative and tune model/request timeouts to avoid worker exhaustion under peak load.

## Memory Pipeline

Memory is on the main chat path:

- Before each turn: load latest `working_context` from SQLite into runtime messages
- After each turn: persist raw/backbone/context snapshots and apply STM compression if threshold exceeded
- Compression and history tables:
  - `raw_messages`
  - `working_context`
  - `compression_events`
