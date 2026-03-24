# Frontend

Current frontend is a static web UI (`index.html`, `styles.css`, `script.js`).

During development, backend serves these files directly:

- `/` -> `index.html`
- `/static/styles.css`
- `/static/script.js`

If you later migrate to a framework (React/Vue), keep this folder as the frontend app root and output build artifacts for backend static serving.
