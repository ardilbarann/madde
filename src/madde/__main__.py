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


if __name__ == "__main__":
    main()
