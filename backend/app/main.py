import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .services import get_repository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def get_allowed_origins():
    raw_origins = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="Manufacturing Optimization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "Frontend build not found. Run `npm install && npm run build` in `frontend/`.",
    }


@app.get("/api/health")
def healthcheck():
    return {
        "status": "ok",
        "frontendBuilt": (FRONTEND_DIST / "index.html").exists(),
    }


@app.get("/api/bootstrap")
def bootstrap():
    repository = get_repository()
    batch_ids = repository.get_batch_ids()

    return {
        "overview": repository.get_overview(),
        "batchIds": batch_ids,
        "defaultBatchId": batch_ids[0] if batch_ids else None,
    }


@app.get("/api/batches/{batch_id}")
def batch_dashboard(batch_id: str):
    repository = get_repository()

    try:
        return repository.get_batch_dashboard(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/batches/{batch_id}/optimization")
def batch_optimization(batch_id: str):
    repository = get_repository()

    try:
        return repository.get_optimization(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        requested_path = FRONTEND_DIST / full_path

        if requested_path.is_file():
            return FileResponse(requested_path)

        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend build not found")
