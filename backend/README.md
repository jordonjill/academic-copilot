# Backend (Academic Copilot)

FastAPI backend for Academic Copilot.

## Architecture

The backend follows a lightweight DDD-style layering under `src/`:

- `domain/`: core state models and domain entities
- `application/`: workflow orchestration and agent logic
- `infrastructure/`: external integrations (LLM/search/MCP/memory/persistence/config)
- `interfaces/`: API adapters (FastAPI routes, DTOs, session/auth glue)

## Directory Layout

```text
backend/
├── main.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── config/
│   ├── agents/
│   └── workflows/
├── data/
└── src/
    ├── domain/
    │   └── state/
    ├── application/
    │   ├── agents/
    │   ├── workflows/
    │   └── graph.py
    ├── infrastructure/
    │   ├── config/
    │   ├── memory/
    │   └── tools/
    └── interfaces/
        └── api/
            └── routes/
```

## Environment Variables

Copy `.env.example` to `.env`:

- `ACCESS_KEY`
- `TAVILY_API_KEY`
- `JINA_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ZOTERO_API_KEY` (optional)

## Run

From repository root:

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend serves frontend static files from `../frontend` at:

- `/` -> `frontend/index.html`
- `/static/*` -> files under `frontend/`

## API Endpoints

- `POST /chat` (Bearer auth)
  - accepts optional `workflow_id`
- `POST /research/start` (Bearer auth)
- `GET /research/status/{session_id}` (Bearer auth)
- `WS /ws/{session_id}?access_key=...`
- `GET /health?model_type=...` (Bearer auth)
- `GET /sessions` (Bearer auth)
- `POST /admin/reload` (Bearer auth)

## Workflow Routing Behavior

- Default proposal routing uses `proposal_v2`.
- Set `PROPOSAL_V2_ROLLBACK=1` to route proposal requests back to legacy `proposal_workflow`.
- `POST /chat` with explicit `workflow_id: "proposal_v2"` also respects the rollback switch and maps to legacy workflow when enabled.

## Runtime Config Reload

- Runtime specs are loaded from:
  - `backend/config/agents/*.yaml`
  - `backend/config/workflows/*.yaml`
- `POST /admin/reload` triggers config revalidation and returns:
  - `config_version`
  - loaded agents/workflows
  - failed objects list

## Confirmation and Dynamic Mode

- Supervisor can suggest workflows and wait for confirmation.
- If confirmation is pending and user sends a new non-confirmation request, suggestion is discarded and mode switches to dynamic execution.
- Dynamic mode enforces same-agent retry caps; when capped with no alternative, runtime marks clarification required.

## Memory Pipeline (MVP-1)

- Persist all raw Human/AI turns to SQLite (`raw_messages`).
- Persist per-turn working context snapshots (`working_context`).
- When STM threshold is exceeded, compress historical context while preserving recent context and log audit entries (`compression_events`).
- Full historical records remain intact even when working context is compressed.
