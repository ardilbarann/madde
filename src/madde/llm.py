"""The two model calls the agent makes, behind an interface a test can fake."""

from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

from madde.grounding import Answer, Failure
from madde.search import Hit

FALLBACK_BETA = "server-side-fallback-2026-07-01"

REWRITE_SYSTEM = """\
You prepare search queries for a corpus of Turkish building regulations: the 2018 seismic code \
(TBDY), the zoning regulation (Planlı Alanlar İmar Yönetmeliği) and the fire protection \
regulation (Binaların Yangından Korunması Hakkında Yönetmelik).

Turn the user's question into one Turkish search query that uses the terms these regulations \
themselves use. The search matches word stems, so the regulation's own vocabulary matters more \
than a fluent sentence. Keep any article reference exactly as written (madde 34, Ek-5/B, 3.2.1). \
If the question is already a good Turkish query, return it unchanged."""

ANSWER_SYSTEM = """\
You answer questions about Turkish building regulations for engineers and architects, who will \
act on what you say. You are given a question and a set of numbered sources, each one article \
or section of a regulation. The sources are the only thing you know about the regulations.

Answer from the sources alone. Split the answer into claims. Each claim is one factual sentence \
backed by one source: copy that source's regulation_id and article exactly, and give a quote \
copied character for character from its text, long enough to show the claim is supported. Quotes \
stay in the original Turkish; the statement and the summary are in the language of the question. \
You may shorten a quote with "..." between two parts of the same source.

Every answer is checked by a program before the user sees it. It rejects a claim whose quote is \
not found in the cited source. It also rejects any number in a statement that is not in that \
statement's own quote, and any number in the summary that no quote contains. This covers every \
digit: measurements, class numbers such as BKS=3, and references to other tables or sections. \
So state numbers the way the source writes them, with the same unit, without converting or \
rounding, choose quotes that contain the numbers you rely on, and leave out a number you \
cannot quote. Numbers the user wrote in the question are always allowed.

If the sources do not answer the question, set found to false, leave claims empty and say in \
the summary what is missing. A partial answer is fine: answer the part the sources support and \
say in the summary which part they do not cover. An honest "not in these sources" is a correct \
result here; an unsupported answer is not."""


class LLMRefused(Exception):
    """The model declined the request (stop_reason 'refusal')."""


class LLM(Protocol):
    def rewrite(self, question: str) -> str: ...

    def answer(self, question: str, hits: list[Hit], feedback: list[Failure]) -> Answer: ...


class _Rewrite(BaseModel):
    query: str = Field(description="Turkish search query.")


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        head = f'<source n="{n}" regulation_id="{h.regulation_id}" article="{h.article}">'
        title = f"{h.heading}\n" if h.heading else ""
        blocks.append(f"{head}\n{title}{h.body}\n</source>")
    return "\n\n".join(blocks)


def answer_prompt(question: str, hits: list[Hit], feedback: list[Failure]) -> str:
    prompt = (
        f"{format_sources(hits)}\n\nQuestion: {question}\n\n"
        "Write the statements and the summary in the language this question is written in."
    )
    if feedback:
        lines = "\n".join(
            f"- {'claim ' + str(f.claim + 1) if f.claim is not None else 'answer'}: {f.reason}"
            for f in feedback
        )
        prompt += (
            "\n\nYour previous answer to this question failed the citation check:\n"
            f"{lines}\n"
            "Answer again. Drop anything the sources cannot support rather than rephrasing it."
        )
    return prompt


class ClaudeLLM:
    def __init__(self, model: str, api_key: str | None = None):
        # With no explicit key the SDK resolves credentials from the environment.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model

    def _parse(self, system: str, prompt: str, schema: type[BaseModel], effort: str):
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            # If a safety classifier declines, the API retries on a fallback model in-call.
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMRefused(getattr(response.stop_details, "category", None) or "refusal")
        if response.parsed_output is None:
            raise ValueError(f"model returned no structured output ({response.stop_reason})")
        return response.parsed_output

    def rewrite(self, question: str) -> str:
        return self._parse(REWRITE_SYSTEM, question, _Rewrite, "low").query

    def answer(self, question: str, hits: list[Hit], feedback: list[Failure]) -> Answer:
        return self._parse(ANSWER_SYSTEM, answer_prompt(question, hits, feedback), Answer, "medium")
