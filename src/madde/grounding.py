"""Deterministic citation check. No model is involved in deciding whether an answer stands.

A claim passes only if
  1. it cites an article that was actually retrieved for this question,
  2. its quote appears in that article's text (whitespace, quote marks and case normalized,
     because the source PDFs carry spacing artifacts such as "Sınıfları ’nın"), and
  3. every number in the claim also appears in its quote, in the article label or in the
     user's own question.
The summary may only use numbers that some claim has already justified.
"""

import re
import unicodedata
from dataclasses import dataclass

from pydantic import BaseModel, Field

from madde.search import Hit

MIN_QUOTE_CHARS = 20  # normalized, without whitespace; a two-word quote proves nothing
_ELLIPSIS = re.compile(r"\[\s*(?:\.{3}|…)\s*\]|\.{3}|…")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_TRANSLATE = str.maketrans(
    {"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"', "–": "-", "—": "-", "­": ""}
)


class Claim(BaseModel):
    statement: str = Field(description="One factual sentence, in the language of the question.")
    regulation_id: str = Field(description="Copied from the source it relies on.")
    article: str = Field(description="Article label copied exactly from that source.")
    quote: str = Field(description="Verbatim Turkish text from that source supporting the claim.")


class Answer(BaseModel):
    found: bool = Field(description="False when the sources do not answer the question.")
    summary: str = Field(description="Short direct answer, or what is missing when found=false.")
    claims: list[Claim]


@dataclass(frozen=True)
class Failure:
    claim: int | None  # index into answer.claims, None for the answer as a whole
    reason: str


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"\s+", "", text)


def numbers(text: str) -> set[str]:
    return {m.group().replace(",", ".") for m in _NUMBER.finditer(text)}


def quote_in_text(quote: str, body: str) -> bool:
    """True when every part of the quote occurs in the body, in order. '...' may skip text."""
    parts = [normalize(p) for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    if sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return False
    haystack, pos = normalize(body), 0
    for part in parts:
        found = haystack.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


def check(answer: Answer, hits: list[Hit], question: str = "") -> list[Failure]:
    """Every reason the answer cannot be shown. An empty list means it is grounded."""
    if not answer.found:
        if answer.claims:
            return [Failure(None, "found is false but claims were given")]
        return []
    if not answer.claims:
        return [Failure(None, "found is true but no claim cites a source")]

    sources = {(h.regulation_id, h.article): h for h in hits}
    failures: list[Failure] = []
    asked = numbers(question)
    justified: set[str] = set(asked)
    for i, claim in enumerate(answer.claims):
        hit = sources.get((claim.regulation_id, claim.article))
        if hit is None:
            failures.append(
                Failure(i, f"{claim.regulation_id} {claim.article} is not among the sources")
            )
            continue
        if not quote_in_text(claim.quote, hit.body):
            failures.append(
                Failure(i, f"quote is not in {claim.regulation_id} {claim.article} verbatim")
            )
            continue
        allowed = (
            asked | numbers(claim.quote) | numbers(claim.article) | numbers(claim.regulation_id)
        )
        justified |= allowed
        loose = numbers(claim.statement) - allowed
        if loose:
            failures.append(Failure(i, f"numbers not in the quote: {', '.join(sorted(loose))}"))
    loose = numbers(answer.summary) - justified
    if loose and not failures:
        failures.append(
            Failure(None, f"summary numbers not in any quote: {', '.join(sorted(loose))}")
        )
    return failures
