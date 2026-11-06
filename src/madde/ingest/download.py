import urllib.request
from pathlib import Path

from madde.ingest.sources import SOURCES

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def download(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in SOURCES:
        path = RAW_DIR / src.filename
        if force or not path.exists():
            req = urllib.request.Request(src.url, headers={"User-Agent": "madde-ingest/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"{src.id}: {src.url} did not return a PDF")
            path.write_bytes(data)
        paths.append(path)
    return paths
