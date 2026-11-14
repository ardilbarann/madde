from dataclasses import asdict
from functools import lru_cache

import anthropic
import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from madde import __version__, agent, db, search
from madde.config import get_settings
from madde.embed import Embedder
from madde.llm import LLM, ClaudeLLM, LLMRefused

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


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    return ClaudeLLM(settings.llm_model, settings.anthropic_api_key)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    regulation: str | None = None


@app.post("/answer")
def answer(body: Question) -> dict:
    """A cited answer, or an explicit not_found / rejected. Nothing unverified is returned."""
    embedder = get_embedder()
    embed = (lambda text: embedder.embed([text])[0]) if embedder else None
    retrieve = agent.db_retriever(get_settings().database_url, embed)
    try:
        return agent.ask(body.question, get_llm(), retrieve, body.regulation)
    except psycopg.Error as exc:
        raise HTTPException(503, "database unavailable") from exc
    except LLMRefused as exc:
        raise HTTPException(422, f"the model declined this question ({exc})") from exc
    except anthropic.AuthenticationError as exc:
        raise HTTPException(503, "model credentials are missing or invalid") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "model rate limit reached, retry later") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"model API error ({exc.status_code})") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(502, "model API unreachable") from exc
