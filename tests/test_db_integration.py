import os

import pytest

from madde import db

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set"),
]


def test_migrate_is_idempotent_and_installs_pgvector():
    db.migrate(DATABASE_URL)
    assert db.migrate(DATABASE_URL) == []
    status = db.check(DATABASE_URL)
    assert status["ok"] and status["pgvector"]


def test_chunk_roundtrip_with_vector_and_fulltext():
    with db.connect(DATABASE_URL) as conn:
        conn.execute(
            "INSERT INTO regulations (id, title, source_url) VALUES ('test-reg', 'Test', 'x')"
            " ON CONFLICT DO NOTHING"
        )
        vec = "[" + ",".join(["0.1"] * 1024) + "]"
        conn.execute(
            "INSERT INTO chunks (regulation_id, article, body, embedding)"
            " VALUES ('test-reg', '1.1', 'deprem yer hareketi düzeyi', %s::vector)"
            " ON CONFLICT (regulation_id, article) DO NOTHING",
            (vec,),
        )
        hit = conn.execute(
            "SELECT article FROM chunks WHERE tsv @@ plainto_tsquery('turkish', 'deprem')"
            " AND regulation_id = 'test-reg'"
        ).fetchone()
        nearest = conn.execute(
            "SELECT article FROM chunks WHERE regulation_id = 'test-reg'"
            " ORDER BY embedding <=> %s::vector LIMIT 1",
            (vec,),
        ).fetchone()
        conn.execute("DELETE FROM regulations WHERE id = 'test-reg'")
    assert hit == ("1.1",)
    assert nearest == ("1.1",)
