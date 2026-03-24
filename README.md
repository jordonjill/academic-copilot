# Academic Copilot (Monorepo)

This repository is split into frontend and backend:

- `frontend/`: web UI assets
- `backend/`: FastAPI service with lightweight DDD-style layering

## Top-level Structure

```text
.
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── backend/
│   ├── main.py
│   ├── pyproject.toml
│   ├── .env.example
│   ├── src/
│   │   ├── domain/
│   │   ├── application/
│   │   ├── infrastructure/
│   │   └── interfaces/
│   └── data/
└── image/
```

## Quick Start

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://127.0.0.1:8000`.

For backend details, see [backend/README.md](backend/README.md).
