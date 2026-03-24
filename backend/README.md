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
- `POST /research/start` (Bearer auth)
- `GET /research/status/{session_id}` (Bearer auth)
- `WS /ws/{session_id}?access_key=...`
- `GET /health?model_type=...` (Bearer auth)
- `GET /sessions` (Bearer auth)
