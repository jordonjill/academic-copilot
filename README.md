# Academic Copilot (Monorepo)

Academic Copilot is split into:

- `backend/`: FastAPI service (supervisor/subagent/workflow runtime, tools, memory)
- `frontend/`: React + TypeScript + Vite workspace UI

## Top-level Structure

```text
.
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── backend/
│   ├── main.py
│   ├── pyproject.toml
│   ├── config/
│   ├── src/
│   ├── tests/
│   ├── scripts/
│   └── data/
└── README.md
```

## Quick Start

### 1) Run backend API

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2) Run frontend (dev mode, optional)

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

### 3) Serve built frontend from backend (optional)

```bash
cd frontend
npm run build
cd ../backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000`.

For details:

- backend: [backend/README.md](backend/README.md)
- frontend: [frontend/README.md](frontend/README.md)
- troubleshooting: see `Troubleshooting` sections in backend/frontend README
