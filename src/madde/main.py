from dataclasses import asdict
from functools import lru_cache

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from madde import __version__, db, search
from madde.config import get_settings
from madde.embed import Embedder

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


@lru_cache
def get_embedder() -> Embedder | None:
    """The query embedder, or None when the optional 'embed' extra is not installed."""
    try:
        from madde.embed import BgeM3

        return BgeM3()
    except ImportError:
        return None


@app.get("/search")
def search_chunks(
    q: str = Query(min_length=2, max_length=500),
    k: int = Query(8, ge=1, le=30),
    regulation: str | None = None,
) -> dict:
    """Hybrid retrieval over the regulation chunks. `modes` says which signals were used."""
    embedder = get_embedder()
    vec = embedder.embed([q])[0] if embedder else None
    try:
        hits = search.hybrid_search(get_settings().database_url, q, vec, k, regulation)
    except psycopg.Error as exc:
        raise HTTPException(503, "database unavailable") from exc
    return {
        "query": q,
        "modes": ["article", "text"] + (["vector"] if vec is not None else []),
        "references": [r.article for r in search.parse_refs(q)],
        "hits": [asdict(h) for h in hits],
    }
