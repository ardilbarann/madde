import argparse

from madde import db
from madde.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m madde")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply SQL migrations")
    ing = sub.add_parser("ingest", help="download, parse and load the regulations")
    ing.add_argument("--force-download", action="store_true")
    ing.add_argument("--dry-run", action="store_true", help="parse and report, do not write")
    emb = sub.add_parser("embed", help="compute bge-m3 embeddings for chunks that lack them")
    emb.add_argument("--redo", action="store_true", help="re-embed everything")
    emb.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto)")
    q = sub.add_parser("nearest", help="semantic nearest chunks for a query (debugging)")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    se = sub.add_parser("search", help="hybrid search: vectors + full-text + article references")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=8)
    se.add_argument("--regulation", default=None, help="tbdy-2018, imar or yangin")
    se.add_argument("--no-vector", action="store_true", help="skip the embedding model")
    ask = sub.add_parser("ask", help="answer a question with verified citations (calls the LLM)")
    ask.add_argument("question")
    ask.add_argument("--regulation", default=None, help="tbdy-2018, imar or yangin")
    ask.add_argument("--no-vector", action="store_true", help="skip the embedding model")
    ask.add_argument("--no-rewrite", action="store_true", help="search with the question as typed")
    args = parser.parse_args()
    url = get_settings().database_url

    if args.cmd == "migrate":
        for name in db.migrate(url):
            print(f"applied {name}")
    elif args.cmd == "ingest":
        from madde.ingest.download import download
        from madde.ingest.load import load, read_pages
        from madde.ingest.parse import parse
        from madde.ingest.sources import SOURCES

        for src, path in zip(SOURCES, download(args.force_download), strict=True):
            chunks = parse(read_pages(path), src.fmt)
            sizes = sorted(len(c.body) for c in chunks)
            print(
                f"{src.id}: {len(chunks)} chunks, median {sizes[len(sizes) // 2]} chars,"
                f" max {sizes[-1]}"
            )
            if not args.dry_run:
                load(url, src, chunks)
    elif args.cmd == "embed":
        from madde.embed import BgeM3, embed_missing

        embed_missing(url, BgeM3(device=args.device), redo=args.redo)
    elif args.cmd == "nearest":
        from madde.embed import BgeM3, nearest

        vec = BgeM3().embed([args.query])[0]
        for reg, article, heading, page, score in nearest(url, vec, args.k):
            print(f"{score:.3f}  {reg} {article}  p{page}  {heading or ''}")
    elif args.cmd == "search":
        from madde.search import hybrid_search

        vec = None
        if not args.no_vector:
            from madde.embed import BgeM3

            vec = BgeM3().embed([args.query])[0]
        for h in hybrid_search(url, args.query, vec, args.k, args.regulation):
            ranks = " ".join(f"{name}#{rank}" for name, rank in h.ranks.items())
            print(f"{h.score:.4f}  {h.regulation_id} {h.article}  p{h.page}  [{ranks}]")
            print(f"        {h.heading or ''}")
    elif args.cmd == "ask":
        from madde import agent
        from madde.llm import ClaudeLLM

        embed = None
        if not args.no_vector:
            from madde.embed import BgeM3

            model = BgeM3()
            embed = lambda text: model.embed([text])[0]  # noqa: E731
        settings = get_settings()
        result = agent.ask(
            args.question,
            ClaudeLLM(settings.llm_model, settings.anthropic_api_key),
            agent.db_retriever(url, embed),
            args.regulation,
            rewrite=not args.no_rewrite,
        )
        print(f"[{result['status']}, {result['attempts']} attempt(s)] search: {result['query']}")
        print(result["summary"])
        for c in result["claims"]:
            print(f'\n- {c["statement"]}\n  {c["regulation_id"]} {c["article"]}: "{c["quote"]}"')
        for reason in result["failures"]:
            print(f"  ! {reason}")


if __name__ == "__main__":
    main()
