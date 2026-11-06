from fastapi.testclient import TestClient

from madde import db
from madde.main import app

client = TestClient(app)


def test_health_does_not_need_database():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_503_when_database_unreachable(monkeypatch):
    monkeypatch.setattr(db, "check", lambda url: {"ok": False, "error": "OperationalError"})
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_ready_requires_pgvector(monkeypatch):
    monkeypatch.setattr(db, "check", lambda url: {"ok": True, "pgvector": None})
    assert client.get("/ready").status_code == 503
