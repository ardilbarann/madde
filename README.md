# Madde

A citation-grounded assistant for Turkish building regulations (TBDY 2018, the zoning regulation, the fire protection regulation).
Every claim in an answer must point to a specific article and quote it. When no article supports an answer, the assistant says so instead of guessing.

**Status:** early. Done: service skeleton, Postgres with pgvector, ingestion of the three regulations into 2,921 citable chunks, local embeddings and hybrid retrieval. The agent and the eval set come next; retrieval quality has not been measured yet.

| Regulation | Source | Layout | Chunks |
|---|---|---|---|
| TBDY 2018 (annex: design rules) | Resmî Gazete, 18.03.2018 | numbered sections (2.3.4.1) | 1,641 |
| Planlı Alanlar İmar Yönetmeliği | mevzuat.gov.tr | MADDE / fıkra / bent | 628 |
| Binaların Yangından Korunması Hakkında Yönetmelik | mevzuat.gov.tr | MADDE / fıkra / bent, annexes | 652 |

Chunks follow the legal structure, so a citation points at something a reader can look up: `tbdy-2018 2.3.4.2`, `imar 4(1)(ll)`, `yangin Ek-5/A`.
Long articles are split by fıkra, then by bent. Each chunk keeps the PDF page it starts on.

Every chunk is embedded locally with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1024-d, multilingual), so Turkish text can be searched in English too.
The model is an optional extra, so the service image and CI stay small:

```bash
uv sync --extra embed
DATABASE_URL=... uv run python -m madde embed            # fills missing vectors (~2 GB model download once)
DATABASE_URL=... uv run python -m madde nearest "imar durum belgesi nedir"
```

## Hybrid retrieval

A question is answered from three ranked lists, merged with weighted reciprocal rank fusion:

| Signal | What it catches | How |
|---|---|---|
| Article reference | "madde 34", "33/2", "Ek-5/B", "TBDY 3.2.1" typed in the question | parsed into the stored label; children included, missing subsections fall back to the parent |
| Dense vectors | paraphrases, English questions | bge-m3 cosine over an HNSW index |
| Full-text | exact legal terms | Postgres Turkish stemmer, terms OR-ed and weighted by inverse document frequency |

An article the user names outranks the other two signals. Each hit reports its rank in every list, so a result can be explained.

```bash
DATABASE_URL=... uv run python -m madde search "yangın merdiveni genişliği en az kaç cm"
curl -G localhost:8000/search --data-urlencode "q=yangın yönetmeliği madde 34" -d k=5
```

`/search` returns `modes`, the signals it actually used. Without the `embed` extra installed (the default image) it runs on references and full-text only.

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
