from fastapi import FastAPI
from fastapi.responses import JSONResponse

from madde import __version__, db
from madde.config import get_settings

app = FastAPI(title="Madde", version=__version__)


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Does not touch the database."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: database reachable and pgvector installed."""
    status = db.check(get_settings().database_url)
    ok = status["ok"] and status.get("pgvector") is not None
    return JSONResponse({"ready": ok, "database": status}, status_code=200 if ok else 503)
