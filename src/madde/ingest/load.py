from pathlib import Path

import pypdf

from madde import db
from madde.ingest.parse import Chunk, parse
from madde.ingest.sources import Source


def read_pages(path: Path) -> list[str]:
    reader = pypdf.PdfReader(path)
    return [(page.extract_text() or "").replace("\x00", "") for page in reader.pages]


def load(database_url: str, src: Source, chunks: list[Chunk]) -> int:
    """Replace one regulation's chunks atomically. Embeddings are filled by a later step."""
    with db.connect(database_url) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO regulations (id, title, source_url, published_on) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title,"
            " source_url = EXCLUDED.source_url, published_on = EXCLUDED.published_on",
            (src.id, src.title, src.url, src.published_on),
        )
        conn.execute("DELETE FROM chunks WHERE regulation_id = %s", (src.id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (regulation_id, article, heading, body, page)"
                " VALUES (%s, %s, %s, %s, %s)",
                [(src.id, c.article, c.heading, c.body, c.page) for c in chunks],
            )
    return len(chunks)
