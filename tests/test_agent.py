import anthropic
import httpx
from fastapi.testclient import TestClient

from madde import agent, main
from madde.grounding import Answer, Claim
from madde.llm import LLMRefused, answer_prompt, format_sources
from madde.search import Hit

BODY = "(2) Yüksek binalarda kaçış yollarının ve merdivenlerin genişliği 120 cm’den az olamaz."
HITS = [Hit(1, "yangin", "33(2)", "Kaçış yolu sayısı ve genişliği", BODY, 40)]
GOOD = Answer(
    found=True,
    summary="En az 120 cm.",
    claims=[
        Claim(
            statement="Yüksek binalarda merdiven genişliği 120 cm'den az olamaz.",
            regulation_id="yangin",
            article="33(2)",
            quote="merdivenlerin genişliği 120 cm’den az olamaz",
        )
    ],
)
BAD = GOOD.model_copy(update={"summary": "En az 150 cm."})
NOT_FOUND = Answer(found=False, summary="Kaynaklar bu soruyu yanıtlamıyor.", claims=[])


class FakeLLM:
    def __init__(self, *answers, rewritten="merdiven genişliği"):
        self.answers = list(answers)
        self.rewritten = rewritten
        self.feedback = []

    def rewrite(self, question):
        return self.rewritten

    def answer(self, question, hits, feedback):
        self.feedback.append(feedback)
        return self.answers.pop(0)


def retriever(hits=HITS):
    seen = []

    def retrieve(query, regulation):
        seen.append((query, regulation))
        return hits

    retrieve.seen = seen
    return retrieve


def test_grounded_answer_is_returned_with_sources():
    retrieve = retriever()
    result = agent.ask("How wide must the stairs be?", FakeLLM(GOOD), retrieve, "yangin")
    assert result["status"] == "answered" and result["attempts"] == 1
    assert result["claims"][0]["article"] == "33(2)"
    assert result["sources"] == [
        {
            "regulation_id": "yangin",
            "article": "33(2)",
            "heading": "Kaçış yolu sayısı ve genişliği",
            "page": 40,
        }
    ]
    assert retrieve.seen == [("merdiven genişliği", "yangin")]  # searched with the rewrite


def test_failed_check_triggers_one_retry_with_the_reasons():
    llm = FakeLLM(BAD, GOOD)
    result = agent.ask("soru", llm, retriever())
    assert result["status"] == "answered" and result["attempts"] == 2
    assert llm.feedback[0] == []
    assert "150" in llm.feedback[1][0].reason


def test_answer_that_fails_twice_is_rejected_without_claims():
    result = agent.ask("soru", FakeLLM(BAD, BAD), retriever())
    assert result["status"] == "rejected" and result["attempts"] == 2
    assert result["claims"] == [] and "150" in result["failures"][0]
    assert "150" not in result["summary"]


def test_model_saying_not_found_is_not_retried():
    result = agent.ask("soru", FakeLLM(NOT_FOUND), retriever())
    assert result["status"] == "not_found" and result["attempts"] == 1


def test_empty_retrieval_skips_the_model():
    llm = FakeLLM()
    result = agent.ask("soru", llm, retriever(hits=[]))
    assert result["status"] == "not_found" and result["attempts"] == 0
    assert llm.feedback == []


def test_rewrite_can_be_disabled_and_empty_rewrite_falls_back():
    retrieve = retriever()
    agent.ask("soru aynen", FakeLLM(GOOD), retrieve, rewrite=False)
    agent.ask("soru aynen", FakeLLM(GOOD, rewritten="  "), retrieve)
    assert [q for q, _ in retrieve.seen] == ["soru aynen", "soru aynen"]


def test_prompt_carries_source_ids_and_feedback():
    text = format_sources(HITS)
    assert '<source n="1" regulation_id="yangin" article="33(2)">' in text and BODY in text
    assert "failed the citation check" not in answer_prompt("q", HITS, [])
    failures = agent.grounding.check(BAD, HITS)
    assert "answer: summary numbers" in answer_prompt("q", HITS, failures)


client = TestClient(main.app)


def _wire(monkeypatch, llm):
    monkeypatch.setattr(main, "get_embedder", lambda: None)
    monkeypatch.setattr(main, "get_llm", lambda: llm)
    monkeypatch.setattr(agent, "db_retriever", lambda url, embed: retriever())


def test_answer_endpoint(monkeypatch):
    _wire(monkeypatch, FakeLLM(GOOD))
    r = client.post("/answer", json={"question": "merdiven genişliği?"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert client.post("/answer", json={"question": "a"}).status_code == 422


def test_answer_endpoint_maps_model_failures(monkeypatch):
    class Refusing(FakeLLM):
        def answer(self, question, hits, feedback):
            raise LLMRefused("cyber")

    class NoKey(FakeLLM):
        def rewrite(self, question):
            response = httpx.Response(401, request=httpx.Request("POST", "https://x"))
            raise anthropic.AuthenticationError("bad key", response=response, body=None)

    _wire(monkeypatch, Refusing())
    assert client.post("/answer", json={"question": "soru?"}).status_code == 422
    _wire(monkeypatch, NoKey())
    assert client.post("/answer", json={"question": "soru?"}).status_code == 503
