"""Split regulation text into citable chunks.

Two layouts exist in the corpus:
- "madde": MADDE 5 - (1) ... (2) ...   (imar, yangın). Long articles are split by
  fıkra "(n)" and, if still long, by bent "a)"; labels look like 5, 5(1), 5(1)(ll).
- "numbered": 2.3.4.2 - ...            (TBDY annex). Each numbered item is a chunk.

Input is a list of page texts, so every chunk keeps the PDF page it starts on.
"""

import re
from dataclasses import dataclass

MAX_CHARS = 2000

TOC_LINE = re.compile(r"\.{5,}")
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
MADDE = re.compile(r"^\s*(GEÇİCİ\s+)?MADDE\s+(\d+)\s*[-–]\s*(.*)$")
# "EK-1", "Ek-1/A Başlık", "Ek - 2 / C". Not "EK-1’de", "Ek-1) 15000", "Ek-1/B, Ek-2/Ç".
ANNEX = re.compile(
    r"^\s*E[Kk]\s*[-–]?\s*(\d+)(?:\s*/\s*([A-ZÇĞİÖŞÜ]))?(?![’'\w)/])\s*(?![,;])[-–:]?\s*(.*)$"
)
PART = re.compile(r"^\s*[A-ZÇĞİÖŞÜ]+\s+BÖLÜM\s*$")
FIKRA = re.compile(r"^\s*\((\d+)\)\s")
BENT = re.compile(r"^\s*([a-zçğıöşü]{1,4})\)\s")
# "2.3.5. Başlık" (heading) or "2.3.4.2 – metin" (paragraph). Anything else is body text.
NUMBERED = re.compile(
    r"^\s*(\d{1,2}[A-Z]?(?:\.\d{1,2}){1,4})(?:(\.)\s+(?=[^\d\s])|\s*[–-]\s*)(.*)$"
)
BOLUM = re.compile(r"^\s*BÖLÜM\s+\d+\s*[–-]")


@dataclass
class Chunk:
    article: str
    heading: str | None
    body: str
    page: int  # 1-based PDF page


def _lines(pages: list[str]):
    for page_no, text in enumerate(pages, start=1):
        lines = text.splitlines()
        if sum(bool(TOC_LINE.search(ln)) for ln in lines) >= 5:
            continue  # table-of-contents page; wrapped titles there have no dot leaders
        for line in lines:
            if TOC_LINE.search(line) or PAGE_NUMBER.match(line) or not line.strip():
                continue
            yield page_no, line.rstrip()


def _join(lines: list[str]) -> str:
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


# ---------------------------------------------------------------- madde layout


def _split(label: str, lines: list[tuple[int, str]], pattern: re.Pattern, fmt: str):
    """Split (page, line) list on pattern; text before the first match stays with label."""
    parts: list[tuple[str, list[tuple[int, str]]]] = [(label, [])]
    used = {label}
    for page, line in lines:
        m = pattern.match(line)
        new = fmt.format(label=label, n=m.group(1)) if m else None
        if new and new not in used:  # repeated numbering (e.g. inside a table) stays as text
            used.add(new)
            parts.append((new, []))
        parts[-1][1].append((page, line))
    return [(lbl, ls) for lbl, ls in parts if ls]


def _emit(label, heading, lines, out, depth=0):
    text = _join([ln for _, ln in lines])
    if len(text) <= MAX_CHARS or depth >= 2:
        out.append(Chunk(label, heading, text, lines[0][0]))
        return
    pattern, fmt = (FIKRA, "{label}({n})") if depth == 0 else (BENT, "{label}({n})")
    parts = _split(label, lines, pattern, fmt)
    if len(parts) == 1:
        out.append(Chunk(label, heading, text, lines[0][0]))
        return
    for lbl, ls in parts:
        _emit(lbl, heading, ls, out, depth + 1)


def _heading(line: str | None) -> str | None:
    """Article titles sit on their own short line ("Amaç", "Kaçış yolları"). Metadata only."""
    if not line:
        return None
    t = line.strip()
    if len(t) < 80 and t[:1].isupper() and not t.endswith((".", ",", ";", ":")):
        return t
    return None


def parse_madde(pages: list[str]) -> list[Chunk]:
    out: list[Chunk] = []
    current: tuple[str, str | None, list] | None = None
    prev_line: str | None = None
    seen: set[str] = set()
    for page, line in _lines(pages):
        if PART.match(line):
            continue
        m = MADDE.match(line)
        a = None if m else ANNEX.match(line)
        label = a and f"Ek-{a.group(1)}" + (f"/{a.group(2)}" if a.group(2) else "")
        if a and label not in seen:
            if current:
                _emit(*current, out)
            seen.add(label)
            current = (label, a.group(3).strip() or None, [(page, line)])
            prev_line = line
            continue
        if m:
            label = ("Geçici " if m.group(1) else "") + m.group(2)
            if label not in seen:
                if current:
                    _emit(*current, out)
                seen.add(label)
                heading = _heading(prev_line)
                head, rest = line[: m.start(3)], m.group(3)
                body = [(page, head)] + ([(page, rest)] if rest else [])
                current = (label, heading, body)
                prev_line = line
                continue
        if current:
            current[2].append((page, line))
        prev_line = line
    if current:
        _emit(*current, out)
    return out


# ------------------------------------------------------------- numbered layout


def _key(label: str) -> tuple[int, ...]:
    """Sort key: chapter, annex letter (0 = main text, A = 1, ...), then subsections.

    Annex 5A comes after the body of chapter 5 and before chapter 6.
    """
    head, *rest = label.split(".")
    m = re.fullmatch(r"(\d+)([A-Z]?)", head)
    annex = ord(m.group(2)) - ord("A") + 1 if m.group(2) else 0
    return (int(m.group(1)), annex, *map(int, rest))


def parse_numbered(pages: list[str]) -> list[Chunk]:
    out: list[Chunk] = []
    headings: dict[str, str] = {}
    seen: set[str] = set()
    current: list | None = None  # [label, heading, page, lines]
    last_key: tuple[int, ...] = (0,)

    def flush():
        if current and current[3]:
            out.append(Chunk(current[0], current[1], _join(current[3]), current[2]))

    for page, line in _lines(pages):
        if BOLUM.match(line):
            continue
        m = NUMBERED.match(line)
        if m and m.group(1) not in seen:
            label, is_heading, rest = m.group(1), bool(m.group(2)), m.group(3).strip()
            key = _key(label)
            # Reject table cells and stray numbers: numbering only moves forward, one chapter at a time.
            first = last_key == (0,)
            if key[0] >= 1 and key > last_key and (first or key[0] <= last_key[0] + 1):
                flush()
                seen.add(label)
                last_key = key
                parent = label.rsplit(".", 1)[0]
                if is_heading and rest:
                    headings[label] = rest
                    current = [label, rest, page, []]
                else:
                    current = [label, headings.get(parent), page, [rest] if rest else []]
                continue
        if current is not None:
            current[3].append(line)
    flush()
    return out


def parse(pages: list[str], fmt: str) -> list[Chunk]:
    if fmt == "madde":
        return parse_madde(pages)
    if fmt == "numbered":
        return parse_numbered(pages)
    raise ValueError(f"unknown format {fmt!r}")
