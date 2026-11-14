import os

import pytest
from fastapi.testclient import TestClient

from madde import db, main, search
from madde.search import ArticleRef, Hit, fuse, hybrid_search, parse_refs

DATABASE_URL = os.environ.get("DATABASE_URL")
integration = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set"),
]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("madde 34 ne diyor", [ArticleRef("34")]),
        ("33 üncü madde", [ArticleRef("33")]),
        ("madde 33/2", [ArticleRef("33(2)")]),
        ("Yangın yönetmeliği 33. maddenin kapsamı", [ArticleRef("33", ("yangin",))]),
        ("imar 5(14)", [ArticleRef("5(14)", ("imar",))]),
        ("geçici madde 2", [ArticleRef("Geçici 2")]),
        ("Ek-5/B tablosu", [ArticleRef("Ek-5/B")]),
        ("ek 5b", [ArticleRef("Ek-5/B")]),
        (
            "9B.1.5 ve 4.5.3.8'e göre",
            [ArticleRef(a, ("tbdy-2018",)) for a in ("9B.1.5", "4.5.3.8")],
        ),
        ("TBDY 3.2 nedir", [ArticleRef("3.2", ("tbdy-2018",))]),
    ],
)
def test_parse_refs(query, expected):
    assert parse_refs(query) == expected


@pytest.mark.parametrize("query", ["kolon aralığı 1.5 m olabilir mi", "0.25 oranı", "ek bina"])
def test_parse_refs_ignores_measurements_and_plain_words(query):
    assert parse_refs(query) == []


def _hit(i: int) -> Hit:
    return Hit(i, "r", str(i), None, "body", 1)


def test_fuse_rewards_agreement_and_named_articles():
    lists = {"vector": [_hit(1), _hit(2), _hit(3)], "text": [_hit(2)]}
    top = fuse(lists, k=3)
    assert [h.chunk_id for h in top] == [2, 1, 3]  # in both lists beats first in one
    assert top[0].ranks == {"vector": 2, "text": 1}

    lists = {"vector": [_hit(1)], "text": [_hit(1)], "article": [_hit(9)]}
    assert fuse(lists, k=1)[0].chunk_id == 9  # the article the user named comes first


def test_ancestor_fallback_labels():
    assert search._with_ancestors("3.2.1") == ["3.2.1", "3.2"]
    assert search._with_ancestors("33(2)") == ["33(2)", "33"]
    assert search._with_ancestors("Ek-5/B") == ["Ek-5/B"]


@pytest.fixture
def corpus():
    db.migrate(DATABASE_URL)
    one_hot = lambda i: "[" + ",".join("1" if j == i else "0" for j in range(1024)) + "]"  # noqa: E731
    rows = [
        ("33", "Kaçış yolu sayısı ve genişliği", "MADDE 33 -"),
        ("33(1)", "Kaçış yolu sayısı ve genişliği", "(1) Toplam çıkış genişliği hesaplanır."),
        ("33(2)", "Kaçış yolu sayısı ve genişliği", "(2) Merdivenlerin genişliği 120 cm’den az "),
        ("34", "YANGIN GÜVENLİK HOLÜ", "MADDE 34 - Yangın güvenlik holleri dumanı engeller."),
        ("3.2", "Deprem tasarım sınıfları", "3.2.1 - Deprem tasarım sınıfları belirlenir."),
    ]
    with db.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM regulations WHERE id = 'srch-test'")
        conn.execute("INSERT INTO regulations VALUES ('srch-test', 'Test', 'x', NULL)")
        for i, (article, heading, body) in enumerate(rows):
            conn.execute(
                "INSERT INTO chunks (regulation_id, article, heading, body, embedding)"
                " VALUES ('srch-test', %s, %s, %s, %s::vector)",
                (article, heading, body, one_hot(i)),
            )
    yield [0.0] * 3 + [1.0] + [0.0] * 1020  # a query vector pointing at article 34
    with db.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM regulations WHERE id = 'srch-test'")


def _articles(hits):
    return [h.article for h in hits]


class TestHybridSearch:
    pytestmark = integration

    def test_turkish_stemming_and_capital_i(self, corpus):
        hits = hybrid_search(DATABASE_URL, "merdivenin genişlikleri", regulation="srch-test")
        assert hits[0].article == "33(2)"
        # "yangın" must match the all-caps heading "YANGIN" despite the dotless i.
        hits = hybrid_search(DATABASE_URL, "yangın holü", regulation="srch-test")
        assert hits[0].article == "34"

    def test_article_reference_lists_children_and_skips_bare_header(self, corpus):
        hits = hybrid_search(DATABASE_URL, "madde 33", regulation="srch-test")
        assert _articles(hits)[:2] == ["33(1)", "33(2)"]  # not the empty "MADDE 33 -" chunk
        assert hits[0].ranks == {"article": 1}

    def test_missing_subsection_falls_back_to_parent(self, corpus):
        hits = hybrid_search(DATABASE_URL, "3.2.1", regulation="srch-test")
        assert _articles(hits)[0] == "3.2"

    def test_named_article_outranks_vector_and_text(self, corpus):
        hits = hybrid_search(
            DATABASE_URL, "madde 33/2 güvenlik holü", corpus, regulation="srch-test"
        )
        assert _articles(hits)[:2] == ["33(2)", "34"]
        assert hits[1].ranks == {"vector": 1, "text": 1}

    def test_no_match_returns_nothing(self, corpus):
        assert hybrid_search(DATABASE_URL, "zzzqqq", regulation="srch-test") == []


client = TestClient(main.app)


def test_search_endpoint_reports_modes_without_embedder(monkeypatch):
    calls = {}

    def fake(url, q, vec, k, regulation):
        calls.update(q=q, vec=vec, k=k, regulation=regulation)
        return [Hit(1, "yangin", "34", "Yangın güvenlik holü", "MADDE 34 - ...", 12, 0.05)]

    monkeypatch.setattr(main, "get_embedder", lambda: None)
    monkeypatch.setattr(search, "hybrid_search", fake)
    r = client.get("/search", params={"q": "madde 34", "k": 3, "regulation": "yangin"})
    assert r.status_code == 200
    data = r.json()
    assert data["modes"] == ["article", "text"]
    assert data["references"] == ["34"]
    assert data["hits"][0]["article"] == "34"
    assert calls == {"q": "madde 34", "vec": None, "k": 3, "regulation": "yangin"}


def test_search_endpoint_uses_embedder_when_present(monkeypatch):
    class Fake:
        name, dim = "fake", 1024

        def embed(self, texts):
            return [[0.5] * 1024 for _ in texts]

    seen = {}
    monkeypatch.setattr(main, "get_embedder", lambda: Fake())
    monkeypatch.setattr(
        search, "hybrid_search", lambda url, q, vec, k, regulation: seen.update(vec=vec) or []
    )
    data = client.get("/search", params={"q": "kaçış yolu"}).json()
    assert data["modes"] == ["article", "text", "vector"]
    assert len(seen["vec"]) == 1024


def test_search_endpoint_validates_input():
    assert client.get("/search", params={"q": "a"}).status_code == 422
    assert client.get("/search", params={"q": "kaçış", "k": 99}).status_code == 422
