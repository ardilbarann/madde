"""Hybrid retrieval: dense vectors, Turkish full-text and explicit article references.

Each of the three produces its own ranked list. The lists are merged with reciprocal rank
fusion, which needs no score calibration between cosine similarity and ts_rank. An article the
user names outright ("madde 34", "Ek-5/B", "3.2.1") outweighs both other signals.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from madde import db
from madde.embed import to_pgvector

RRF_K = 60
WEIGHTS = {"article": 3.0, "vector": 1.0, "text": 1.0}
CANDIDATES = 30  # per list, before fusion
MIN_BODY = 25  # shorter chunks are bare "MADDE 33 -" headers whose text lives in their children

_COLS = "c.id, c.regulation_id, c.article, c.heading, c.body, c.page"
_BASE = f"length(c.body) >= {MIN_BODY}"


@dataclass(frozen=True)
class ArticleRef:
    """An article label as stored, plus the regulations it may belong to (None = any)."""

    article: str
    regulations: tuple[str, ...] | None = None


@dataclass
class Hit:
    chunk_id: int
    regulation_id: str
    article: str
    heading: str | None
    body: str
    page: int | None
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank in each list it was in


def _lower_tr(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


_HINTS = {
    "tbdy-2018": re.compile(r"\btbdy\b|deprem yönetmeli"),
    "imar": re.compile(r"\bimar\b"),
    "yangin": re.compile(r"\byang[ıi]n|\bbyky\b"),
}
_ORDINAL = r"(?:\.|\s*['’]?\s*(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü))"
_FIKRA = r"(?:\s*/\s*(\d{1,2})|\s*\((\d{1,2})\))?"
_GECICI = re.compile(r"\bgeçici\s+(?:madde\s+)?(\d{1,2})" + _FIKRA)
_EK = re.compile(r"\bek[\s\-]*(\d{1,2})(?:\s*/\s*([a-zç])\b|([a-zç])\b)?")
_MADDE = re.compile(r"\bmadde\s*(\d{1,3})" + _FIKRA)
_MADDE_REV = re.compile(r"\b(\d{1,3})" + _ORDINAL + r"\s*madde")
_BARE = re.compile(r"(?<![\w.])(\d{1,3})\((\d{1,2})\)")
_NUMBERED = re.compile(r"(?<![\w.,])(\d{1,2}[A-D]?(?:\.\d{1,2}){1,4})(?![\w])(?!\.\d)")


def parse_refs(query: str) -> list[ArticleRef]:
    """Find article references the user typed, in the label format the chunks use."""
    low = _lower_tr(query)
    hinted = [reg for reg, pat in _HINTS.items() if pat.search(low)]
    madde_regs = tuple(r for r in hinted if r != "tbdy-2018") or None
    refs: list[ArticleRef] = []

    def add(article: str, regs: tuple[str, ...] | None) -> None:
        ref = ArticleRef(article, regs)
        if ref not in refs:
            refs.append(ref)

    def with_fikra(base: str, m: re.Match, first_group: int) -> str:
        fikra = m.group(first_group) or m.group(first_group + 1)
        return f"{base}({fikra})" if fikra else base

    for m in _GECICI.finditer(low):
        add(with_fikra(f"Geçici {m.group(1)}", m, 2), madde_regs)
    low = _GECICI.sub(" ", low)
    for m in _EK.finditer(low):
        letter = m.group(2) or m.group(3)
        add(f"Ek-{m.group(1)}" + (f"/{letter.upper()}" if letter else ""), None)
    for m in _MADDE.finditer(low):
        add(with_fikra(m.group(1), m, 2), madde_regs)
    for m in _MADDE_REV.finditer(low):
        add(m.group(1), madde_regs)
    for m in _BARE.finditer(low):
        add(f"{m.group(1)}({m.group(2)})", madde_regs)
    # Section numbers keep their capital annex letter (9B.1.5), so match on the original text.
    for m in _NUMBERED.finditer(query):
        label = m.group(1)
        # "1.5" is more often a measurement than section 1.5; accept it only with a TBDY cue.
        if label.count(".") >= 2 or "tbdy-2018" in hinted:
            add(label, ("tbdy-2018",))
    return refs


def fuse(lists: dict[str, list[Hit]], k: int) -> list[Hit]:
    """Weighted reciprocal rank fusion over named ranked lists."""
    merged: dict[int, Hit] = {}
    for name, hits in lists.items():
        for rank, hit in enumerate(hits, start=1):
            kept = merged.setdefault(hit.chunk_id, hit)
            kept.ranks[name] = rank
            kept.score += WEIGHTS[name] / (RRF_K + rank)
    return sorted(merged.values(), key=lambda h: (-h.score, h.chunk_id))[:k]


def _hits(rows: list[tuple]) -> list[Hit]:
    return [Hit(*row) for row in rows]


def _by_article(conn, refs: list[ArticleRef], regulation: str | None) -> list[Hit]:
    out: list[Hit] = []
    seen: set[int] = set()
    for ref in refs:
        rows: list[tuple] = []
        # Short subsections are stored inside their parent chunk (TBDY 3.2.1 lives in 3.2),
        # so a reference with no chunk of its own falls back to the closest ancestor.
        for label in _with_ancestors(ref.article):
            rows = _article_rows(conn, label, ref.regulations, regulation)
            if rows:
                break
        for hit in _hits(rows):
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                out.append(hit)
    return out[:CANDIDATES]


def _with_ancestors(article: str) -> list[str]:
    """'3.2.1' -> ['3.2.1', '3.2'];  '33(2)' -> ['33(2)', '33'].  Never up to a bare chapter."""
    labels = [article]
    while True:
        parent = re.sub(r"(\(\w+\)|\.\d+)$", "", labels[-1])
        if parent == labels[-1] or ("." in article and "." not in parent):
            return labels
        labels.append(parent)


def _article_rows(
    conn, article: str, regulations: tuple[str, ...] | None, regulation: str | None
) -> list[tuple]:
    # Children of a section are "3.2.1" under "3.2"; children of a madde are "33(1)" under "33".
    child = article + (".%" if "." in article else "(%")
    return conn.execute(
        f"SELECT {_COLS} FROM chunks c WHERE {_BASE}"
        " AND (c.article = %(a)s OR c.article LIKE %(child)s)"
        " AND (%(regs)s::text[] IS NULL OR c.regulation_id = ANY(%(regs)s))"
        " AND (%(reg)s::text IS NULL OR c.regulation_id = %(reg)s)"
        " ORDER BY (c.article = %(a)s) DESC, c.id LIMIT %(n)s",
        {
            "a": article,
            "child": child,
            "regs": list(regulations) if regulations else None,
            "reg": regulation,
            "n": CANDIDATES,
        },
    ).fetchall()


def _by_text(conn, query: str, regulation: str | None) -> list[Hit]:
    # OR the stemmed query terms: a question never matches an article on every word.
    # Postgres ranking has no notion of rare terms, so "durum" would count as much as "imar".
    # Weight each matched term by its inverse document frequency; ts_rank_cd breaks ties.
    stats = conn.execute(
        "SELECT t.lexeme, (SELECT count(*) FROM chunks c WHERE c.tsv @@"
        "   to_tsquery('simple', '''' || replace(t.lexeme, '''', '''''') || '''')),"
        " (SELECT count(*) FROM chunks)"
        " FROM unnest(to_tsvector('turkish', translate(%s, 'Iİ', 'ıi'))) t",
        (query,),
    ).fetchall()
    terms = [(lex, math.log(1 + (total - df + 0.5) / (df + 0.5))) for lex, df, total in stats if df]
    if not terms:
        return []
    tsquery = " | ".join("'" + lex.replace("'", "''") + "'" for lex, _ in terms)
    rows = conn.execute(
        f"SELECT {_COLS} FROM chunks c, to_tsquery('simple', %(q)s) tq"
        f" WHERE {_BASE} AND c.tsv @@ tq"
        " AND (%(reg)s::text IS NULL OR c.regulation_id = %(reg)s)"
        " ORDER BY (SELECT sum(t.idf) FROM unnest(%(lex)s::text[], %(idf)s::float8[])"
        "           AS t(lex, idf) WHERE t.lex = ANY(tsvector_to_array(c.tsv))) DESC,"
        "   ts_rank_cd(c.tsv, tq, 1) DESC, c.id LIMIT %(n)s",
        {
            "q": tsquery,
            "lex": [lex for lex, _ in terms],
            "idf": [idf for _, idf in terms],
            "reg": regulation,
            "n": CANDIDATES,
        },
    ).fetchall()
    return _hits(rows)


def _by_vector(conn, query_vec: Sequence[float], regulation: str | None) -> list[Hit]:
    # HNSW filters after the index scan; keep scanning until the regulation filter is satisfied.
    conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    rows = conn.execute(
        f"SELECT {_COLS} FROM chunks c WHERE {_BASE} AND c.embedding IS NOT NULL"
        " AND (%(reg)s::text IS NULL OR c.regulation_id = %(reg)s)"
        " ORDER BY c.embedding <=> %(v)s::vector LIMIT %(n)s",
        {"v": to_pgvector(query_vec), "reg": regulation, "n": CANDIDATES},
    ).fetchall()
    return _hits(rows)


def hybrid_search(
    database_url: str,
    query: str,
    query_vec: Sequence[float] | None = None,
    k: int = 8,
    regulation: str | None = None,
) -> list[Hit]:
    """Top-k chunks for a question. Without `query_vec` it runs on text and references only."""
    lists: dict[str, list[Hit]] = {}
    with db.connect(database_url) as conn:
        refs = parse_refs(query)
        if refs:
            lists["article"] = _by_article(conn, refs, regulation)
        if query_vec is not None:
            lists["vector"] = _by_vector(conn, query_vec, regulation)
        lists["text"] = _by_text(conn, query, regulation)
    return fuse(lists, k)
