# Manufacturing Optimization System

This project is now fully centered on a React + FastAPI architecture:

- `backend/`: FastAPI API for model loading, batch analytics, alerts, and optimization.
- `frontend/`: React + Vite dashboard for the manufacturing control room.
- `Dockerfile`: production image that builds the frontend and serves everything through FastAPI.
- `Procfile`: simple process entry for platforms that launch a single web command.

## Local development

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run build
npm run dev
```

The React app uses the same host by default. For local frontend-only development, you can point it to a separate backend by copying `frontend/.env.example` to `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

For production, build the frontend and let FastAPI serve `frontend/dist`:

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Deployment

Docker:

```bash
docker build -t manufacturing-optimization-system .
docker run -p 8000:8000 manufacturing-optimization-system
```

Single-service platforms:

- Render, Railway, Fly.io, or Heroku-style platforms can use the root `Procfile`.
- Set `CORS_ALLOW_ORIGINS` if you host the frontend separately.
- Keep the `.pkl` model files out of git and let the backend download them from Google Drive when needed.
