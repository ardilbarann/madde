"""The answering graph: rewrite -> retrieve -> generate -> verify, with one corrective retry.

    rewrite ──> retrieve ──> generate ──> verify ──(grounded or out of attempts)──> END
                   │            ▲            │
                   │            └──(failed)──┘
                   └──(nothing retrieved)──> END

`verify` is deterministic (madde.grounding). An answer that still fails after the retry is
returned as "rejected" with the reasons and none of its claims.
"""

from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from madde import grounding
from madde.grounding import Answer, Failure
from madde.llm import LLM
from madde.search import Hit

MAX_ATTEMPTS = 2
Retriever = Callable[[str, str | None], list[Hit]]  # (query, regulation) -> hits
Embed = Callable[[str], Sequence[float]] | None


class State(TypedDict, total=False):
    question: str
    regulation: str | None
    query: str
    hits: list[Hit]
    answer: Answer | None
    failures: list[Failure]
    attempts: int


class Result(TypedDict):
    status: Literal["answered", "not_found", "rejected"]
    summary: str
    claims: list[dict]
    failures: list[str]
    attempts: int
    query: str
    sources: list[dict]


def build_graph(llm: LLM, retrieve: Retriever, rewrite: bool = True):
    def rewrite_node(state: State) -> State:
        query = llm.rewrite(state["question"]).strip() if rewrite else ""
        return {"query": query or state["question"]}

    def retrieve_node(state: State) -> State:
        return {"hits": retrieve(state["query"], state.get("regulation"))}

    def generate_node(state: State) -> State:
        answer = llm.answer(state["question"], state["hits"], state.get("failures", []))
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    def verify_node(state: State) -> State:
        return {"failures": grounding.check(state["answer"], state["hits"], state["question"])}

    def after_retrieve(state: State) -> str:
        return "generate" if state["hits"] else END

    def after_verify(state: State) -> str:
        return "generate" if state["failures"] and state["attempts"] < MAX_ATTEMPTS else END

    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, ["generate", END])
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, ["generate", END])
    return graph.compile()


def _source(hit: Hit) -> dict:
    return {
        "regulation_id": hit.regulation_id,
        "article": hit.article,
        "heading": hit.heading,
        "page": hit.page,
    }


def to_result(state: State) -> Result:
    answer, failures = state.get("answer"), state.get("failures", [])
    base = {
        "attempts": state.get("attempts", 0),
        "query": state["query"],
        "sources": [_source(h) for h in state.get("hits", [])],
    }
    if answer is None:
        return Result(
            status="not_found",
            summary="No article in the corpus matched the question.",
            claims=[],
            failures=[],
            **base,
        )
    if failures:
        return Result(
            status="rejected",
            summary="The answer could not be verified against the regulation text.",
            claims=[],
            failures=[f.reason for f in failures],
            **base,
        )
    return Result(
        status="answered" if answer.found else "not_found",
        summary=answer.summary,
        claims=[c.model_dump() for c in answer.claims],
        failures=[],
        **base,
    )


def ask(
    question: str,
    llm: LLM,
    retrieve: Retriever,
    regulation: str | None = None,
    rewrite: bool = True,
) -> Result:
    graph = build_graph(llm, retrieve, rewrite)
    return to_result(graph.invoke({"question": question, "regulation": regulation}))


def db_retriever(database_url: str, embed: Embed = None, k: int = 8) -> Retriever:
    """Hybrid search bound to a database; `embed` adds the vector signal when available."""
    from madde.search import hybrid_search

    def retrieve(query: str, regulation: str | None) -> list[Hit]:
        vec = embed(query) if embed else None
        return hybrid_search(database_url, query, vec, k, regulation)

    return retrieve
