"""Dense embeddings with BAAI/bge-m3 (1024-d, multilingual, no query instruction needed).

The model is an optional dependency (`uv sync --extra embed`) so the API image and CI stay
small. Anything that only needs the Embedder interface can use a fake in tests.
"""

from collections.abc import Sequence
from typing import Protocol

from madde import db

MODEL_NAME = "BAAI/bge-m3"
MAX_TOKENS = 1024  # chunks are article-sized; the long tail is tables, truncation is acceptable


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class BgeM3:
    name = MODEL_NAME
    dim = 1024

    def __init__(self, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.model.max_seq_length = MAX_TOKENS
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        return vecs.tolist()


def passage_text(title: str, article: str, heading: str | None, body: str) -> str:
    """What gets embedded: regulation and article context first, then the text itself."""
    head = f"{title} | {article}" + (f" | {heading}" if heading else "")
    return f"{head}\n{body}"


def to_pgvector(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def embed_missing(
    database_url: str,
    embedder: Embedder,
    batch: int = 64,
    redo: bool = False,
    regulation_id: str | None = None,
) -> int:
    """Fill embeddings for chunks that have none (or were made by another model)."""
    done = 0
    where = "TRUE" if redo else "(c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM %(m)s)"
    if regulation_id:
        where += " AND c.regulation_id = %(r)s"
    with db.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT c.id, r.title, c.article, c.heading, c.body FROM chunks c"
            f" JOIN regulations r ON r.id = c.regulation_id WHERE {where} ORDER BY c.id",
            {"m": embedder.name, "r": regulation_id},
        ).fetchall()
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            vecs = embedder.embed([passage_text(*r[1:]) for r in part])
            if any(len(v) != embedder.dim for v in vecs):
                raise ValueError("embedding dimension does not match the embedder")
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
                    [
                        (to_pgvector(v), embedder.name, r[0])
                        for r, v in zip(part, vecs, strict=True)
                    ],
                )
            done += len(part)
            print(f"embedded {done}/{len(rows)}", flush=True)
    return done


def nearest(database_url: str, query_vec: Sequence[float], k: int = 5) -> list[tuple]:
    with db.connect(database_url) as conn:
        return conn.execute(
            "SELECT regulation_id, article, heading, page, 1 - (embedding <=> %s::vector) AS score"
            " FROM chunks WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s",
            (to_pgvector(query_vec), to_pgvector(query_vec), k),
        ).fetchall()
