import hashlib
import os

import pytest

from madde import db
from madde.embed import embed_missing, nearest, passage_text, to_pgvector

DATABASE_URL = os.environ.get("DATABASE_URL")


class FakeEmbedder:
    """Deterministic 1024-d vectors from a hash; same text -> same vector."""

    name = "fake"
    dim = 1024

    def embed(self, texts):
        out = []
        for t in texts:
            seed = hashlib.sha256(t.encode()).digest()
            out.append([(seed[i % 32] - 128) / 128 for i in range(self.dim)])
        return out


def test_passage_text_puts_context_first():
    t = passage_text("İmar Yönetmeliği", "4(1)(ll)", "Tanımlar", "İmar durum belgesi: ...")
    assert t.splitlines()[0] == "İmar Yönetmeliği | 4(1)(ll) | Tanımlar"
    assert passage_text("X", "1", None, "b").splitlines()[0] == "X | 1"


def test_to_pgvector_format():
    assert to_pgvector([0.5, -1]) == "[0.500000,-1.000000]"


@pytest.mark.integration
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_embed_missing_fills_only_missing_and_records_model():
    db.migrate(DATABASE_URL)
    with db.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM regulations WHERE id = 'emb-test'")
        conn.execute("INSERT INTO regulations VALUES ('emb-test', 'Test Yön.', 'x', NULL)")
        for art, body in [("1", "deprem"), ("2", "yangın")]:
            conn.execute(
                "INSERT INTO chunks (regulation_id, article, body) VALUES ('emb-test', %s, %s)",
                (art, body),
            )
    fake = FakeEmbedder()
    try:
        assert embed_missing(DATABASE_URL, fake, regulation_id="emb-test") == 2
        assert embed_missing(DATABASE_URL, fake, regulation_id="emb-test") == 0  # idempotent
        target = fake.embed([passage_text("Test Yön.", "2", None, "yangın")])[0]
        top = nearest(DATABASE_URL, target, k=1)[0]
        assert top[:2] == ("emb-test", "2") and top[4] > 0.999
    finally:
        with db.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM regulations WHERE id = 'emb-test'")
