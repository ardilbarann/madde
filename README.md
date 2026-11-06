# Madde

A citation-grounded assistant for Turkish building regulations (TBDY 2018, the zoning regulation, the fire protection regulation).
Every claim in an answer must point to a specific article and quote it. When no article supports an answer, the assistant says so instead of guessing.

**Status:** early. Done: service skeleton, Postgres with pgvector, and ingestion of the three regulations into 2,921 citable chunks. Retrieval, the agent and the eval set come next.

| Regulation | Source | Layout | Chunks |
|---|---|---|---|
| TBDY 2018 (annex: design rules) | Resmî Gazete, 18.03.2018 | numbered sections (2.3.4.1) | 1,641 |
| Planlı Alanlar İmar Yönetmeliği | mevzuat.gov.tr | MADDE / fıkra / bent | 628 |
| Binaların Yangından Korunması Hakkında Yönetmelik | mevzuat.gov.tr | MADDE / fıkra / bent, annexes | 652 |

Chunks follow the legal structure, so a citation points at something a reader can look up: `tbdy-2018 2.3.4.2`, `imar 4(1)(ll)`, `yangin Ek-5/A`.
Long articles are split by fıkra, then by bent. Each chunk keeps the PDF page it starts on.

## Run locally

```bash
docker compose up --build
curl localhost:8000/ready
```

Load the regulations (downloads the PDFs into `data/raw/`, which is not committed):

```bash
DATABASE_URL=postgresql://madde:madde@localhost:5442/madde uv run python -m madde ingest
```

Tests:

```bash
docker compose up -d db
DATABASE_URL=postgresql://madde:madde@localhost:5442/madde uv run pytest
```

## Planned design

- Article-level chunks with hybrid retrieval (pgvector + Postgres full-text, exact article-number match)
- LangGraph agent producing structured answers: claim, article id, quote
- Deterministic citation check: a quote must appear verbatim in the cited article, or the answer is rejected
- Eval set of 60 questions (10 out of scope) with retrieval recall, citation accuracy and refusal rate reported here
- Langfuse tracing, Next.js UI, AWS deployment with Terraform

## Source texts

Regulations are published in the Resmî Gazete and are official texts not covered by copyright (FSEK art. 31).
The repository does not ship the PDFs; an ingest script will download them from the official sources.
Answers are not a substitute for the official text in engineering decisions.

## License

MIT
