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

- `ACCESS_KEY`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `JINA_API_KEY`
- `GOOGLE_API_KEY`
- `ZOTERO_API_KEY` (optional)

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

- `POST /chat` (Bearer auth)
  - request: `message`, optional `workflow_id`, optional `session_id`, optional `user_id`
- `GET /health` (Bearer auth)
- `POST /admin/reload` (Bearer auth)
  - reload tools + runtime config
- `POST /admin/reload-runtime` (Bearer auth)
- `POST /admin/reload-tools` (Bearer auth)

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

## Memory Pipeline

Memory is on the main chat path:

- Before each turn: load latest `working_context` from SQLite into runtime messages
- After each turn: persist raw/backbone/context snapshots and apply STM compression if threshold exceeded
- Compression and history tables:
  - `raw_messages`
  - `working_context`
  - `compression_events`
