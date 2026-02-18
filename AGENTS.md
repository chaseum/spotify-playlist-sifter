# AGENTS.md

## Project

- Purpose: backend-first Spotify Web API integration project.
- Stack: Python 3.12 + FastAPI + SQL database (Postgres preferred; SQLite allowed for early dev).
- Frontend: thin testing shell served via FastAPI StaticFiles.
- Location: app/web/
- Contains index.html, app.js, styles.css
- UI only — no Spotify API logic.
- Do not assume `index.html` or `app.js` exist at repo root.

## Workspace discovery (avoid roadblocks)

Before planning or patching:

1. Print the repo tree (2–3 levels) and identify the actual entrypoints:
   - Look for: pyproject.toml / requirements.txt / Dockerfile / docker-compose.yml / app/ / src/ / apps/
   - Also detect: package.json (only if a frontend exists)
2. If backend files are missing, propose a minimal scaffold (FastAPI app + config + one route) and create it.

## Spotify constraints (must follow)

- OAuth: Authorization Code with PKCE. No implicit flow.
- Redirect URIs: use HTTPS except loopback; use explicit 127.0.0.1; localhost aliases are not allowed.
- Feb 2026 changes: playlist endpoints use /items (not /tracks); library write uses /me/library; other field/behavior changes apply.

## Setup commands (local)

- Create venv (one-time): `python -m venv .venv`
- Install deps: `pip install -r requirements.txt`
- Run API: `uvicorn app.main:app --reload --port 8000`
- Activate (Windows PowerShell, if needed): `.venv\\Scripts\\Activate.ps1`

## Test/lint commands

- Tests: `pytest -q`
- Lint/format (if configured): `ruff check .` and `ruff format .`

## Edit rules

- Minimal diffs. No refactors unless asked.
- Touch the smallest file set possible; explain why if >3 files.
- Add tests for new behavior (happy path + failure).
- Never log tokens/PII. Redact secrets in logs.

## Output requirements

- Provide patch/diff only for code changes.
- After patches: list files changed + why, and show test command results.

## Frontend inspiration references

- If a /docs/frontend_inspo/ folder exists, treat it as design reference only.
- You may mirror CSS tokens, spacing, and accessibility patterns later, but do not require those files to exist.

## Architecture rule: Spotify API ownership

- The backend owns ALL Spotify Web API communication.
- Frontend (`app/web/app.js`) must NEVER call `https://api.spotify.com/*` directly.
- Frontend only calls internal routes such as:
  - `/api/me`
  - `/api/playlists`
  - `/api/library`

- Spotify access tokens and refresh logic live ONLY in:
  - `app/services/spotify_client.py`
  - `app/services/spotify_oauth.py`

Frontend responsibilities:

- Redirect user to `/auth/spotify/login`
- Display connection state
- Call backend API routes

Backend responsibilities:

- Build authorize URL (PKCE)
- Exchange code for tokens
- Refresh tokens automatically
- Proxy Spotify Web API requests

Reason:

- Prevent token exposure in browser
- Keep refresh logic centralized
- Align with backend-first architecture
