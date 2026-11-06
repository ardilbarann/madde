from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, connect_timeout=3)


def migrate(database_url: str) -> list[str]:
    """Apply every migrations/*.sql file not yet recorded, in name order."""
    applied: list[str] = []
    with connect(database_url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


def check(database_url: str) -> dict:
    """Return database reachability and pgvector status, never raise."""
    try:
        with connect(database_url) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        return {"ok": True, "pgvector": row[0] if row else None}
    except psycopg.Error as exc:
        return {"ok": False, "error": type(exc).__name__}
