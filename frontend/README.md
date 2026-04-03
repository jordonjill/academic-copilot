# Academic Copilot Frontend

React + TypeScript + Vite frontend for Academic Copilot backend.

## Features

- Session list and session switching
- Direct mode / workflow mode selection
- Multi-turn chat in the same `session_id`
- Realtime streaming chat via `POST /chat/stream` (SSE)
- Runtime panel (`mode/workflow/node/step/loop/tool_budget`)
- Export panel (DOCX/PDF path from artifacts when available)
- Long timeout support for long-running workflows

## Quick Start

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Default UI URL: `http://127.0.0.1:5173`

## Build

```bash
cd frontend
npm run build
```

- Type check: `tsc --noEmit`
- Bundle output: `dist/` (served by backend at `/` and `/assets/*` when present)

## Environment

- `VITE_API_BASE_URL` backend base URL
- `VITE_ACCESS_KEY` backend access key used in `Authorization: Bearer ...`
- `VITE_REQUEST_TIMEOUT_MS` request timeout (ms), should be larger than backend `CHAT_TURN_TIMEOUT_SECONDS`
- `VITE_DEFAULT_USER_ID` default user id for UI

## Troubleshooting

- If UI requests timeout on long workflows, increase `VITE_REQUEST_TIMEOUT_MS`.
- Keep `VITE_REQUEST_TIMEOUT_MS` greater than backend `CHAT_TURN_TIMEOUT_SECONDS` to avoid frontend aborting earlier than backend.
- If stream connects but no final result arrives, check backend logs for:
  - upstream model retries
  - tool rate-limit events (for example arXiv 429)
  - turn timeout (`chat.turn.timeout`)
