"""Regulatory corpus: PDF -> text -> paragraph-level chunks with stable, human-readable ids.

Chunk ids are derived only from the document structure (article / paragraph / Tz numbers), so
they stay identical across runs and re-indexing, e.g. ``crr.art92.1``, ``marisk.btr3.1.tz4``,
``lcr30.40``, ``ebagl.p115``. Long paragraphs are split into ``#2``, ``#3`` parts.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kapitalkompass.config import ROOT, load_config

NOISE = [
    re.compile(p)
    for p in (
        r"^\d+ / \d+$",
        r"^Seite \d+ von \d+$",
        r"^0201\dR\d{4} — EN —",
        r"^0201\dL\d{4} — EN —",
        r"^[▼►][A-Z0-9 ►]*$",
        r"^ELI: http",
        r"^\d{1,3}$",  # page numbers (MaRisk Tz handled before noise removal)
        r"^FINAL REPORT ON GUIDELINES ON THE MANAGEMENT",
        r"^RISK ARISING FROM NON-TRADING BOOK ACTIVITIES$",
        r"^This standard has been integrated",
        r"^Official Journal of the European Union$",
    )
]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    ref: str  # citation label for humans, e.g. "Art. 92(1) CRR"
    text: str


def pdf_text(path: Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, check=True
    )
    return out.stdout.decode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(lines: list[str], keep_numbers: bool = False) -> str:
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if any(p.match(s) for p in NOISE) and not (keep_numbers and s.isdigit()):
            continue
        kept.append(s)
    txt = unicodedata.normalize("NFKC", " ".join(kept))
    txt = re.sub(r"(\w)- (\w)", r"\1\2", txt)  # de-hyphenate line breaks
    txt = txt.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    return re.sub(r"\s+", " ", txt).strip()


def _split(cid: str, doc: str, ref: str, text: str, max_chars: int) -> list[Chunk]:
    if len(text) <= max_chars:
        return [Chunk(cid, doc, ref, text)]
    parts: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.;:])\s+", text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return [Chunk(cid if i == 0 else f"{cid}#{i + 1}", doc, ref, p) for i, p in enumerate(parts)]


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen: dict[str, int] = {}
    out = []
    for c in chunks:
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        out.append(c if n == 0 else Chunk(f"{c.id}~{n + 1}", c.doc, c.ref, c.text))
    return out


def parse_eu_articles(text: str, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Selected articles; one chunk per numbered paragraph (``1.``, ``1a.``)."""
    lines = text.splitlines()
    head = re.compile(r"^Article (\d+[a-z]*)$")
    out: list[tuple[str, str, str]] = []
    for art in doc["articles"]:
        start = None
        for i, ln in enumerate(lines):
            if ln.strip() == f"Article {art}":
                nxt = next((x.strip() for x in lines[i + 1 : i + 6] if x.strip()), "")
                if nxt and not nxt.startswith("Article"):
                    start = i
                    break
        if start is None:
            raise ValueError(f"{doc['id']}: Article {art} not found")
        end = next(
            (j for j in range(start + 1, len(lines)) if head.match(lines[j].strip())), len(lines)
        )
        body = lines[start + 1 : end]
        title = next(x.strip() for x in body if x.strip())
        paras: list[tuple[str, list[str]]] = [("0", [])]
        for ln in body:
            m = re.match(r"^(\d+[a-z]?)\.$", ln.strip())
            if m:
                paras.append((m.group(1), []))
            else:
                paras[-1][1].append(ln)
        for num, plines in paras:
            t = _clean(plines)
            if num == "0":
                if len(paras) > 1 or not t:
                    continue  # only the title before paragraph 1
                ref = f"Art. {art} {doc['label']}"
                cid = f"{doc['id']}.art{art}"
            else:
                ref = f"Art. {art}({num}) {doc['label']}"
                cid = f"{doc['id']}.art{art}.{num}"
            out.append((cid, ref, f"{ref} – {title}: {t}"))
    return out


def _monotonic(
    lines: list[str], pat: re.Pattern[str], max_jump: int = 3, first: int = 1
) -> list[tuple[int, list[str]]]:
    """Split at numbered paragraph starts, accepting a number only if it is last+1..last+jump."""
    paras: list[tuple[int, list[str]]] = []
    last = first - 1
    for ln in lines:
        m = pat.match(ln.strip())
        if m and last < int(m.group(1)) <= last + max_jump:
            last = int(m.group(1))
            rest = ln.strip()[m.end() :]
            paras.append((last, [rest] if rest else []))
        elif paras:
            paras[-1][1].append(ln)
    return paras


def _start_index(lines: list[str], pattern: str, occurrence: int) -> int:
    hits = [i for i, ln in enumerate(lines) if re.match(pattern, ln.strip())]
    return hits[occurrence - 1]


def parse_numbered(text: str, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    s = _start_index(lines, doc["start"], int(doc.get("occurrence", 1))) + 1
    e = _start_index(lines, doc["end"], 1) if "end" in doc else len(lines)
    if e <= s:
        e = next(i for i in range(s, len(lines)) if re.match(doc["end"], lines[i].strip()))
    annex_at: list[tuple[int, str]] = []
    if "annex" in doc:
        apat = re.compile(doc["annex"])
        annex_at = [(i, m.group(1)) for i in range(s, e) if (m := apat.match(lines[i].strip()))]
    body_end = annex_at[0][0] if annex_at else e
    out = []
    first = int(doc.get("first_para", 1))
    for n, plines in _monotonic(lines[s:body_end], re.compile(doc["para"]), first=first):
        ref = f"{doc['label']}, Rn. {n}" if doc["id"] == "ebagl" else f"{doc['label']}, para {n}"
        out.append((f"{doc['id']}.p{n}", ref, f"{ref}: {_clean(plines)}"))
    for k, (i, num) in enumerate(annex_at):
        j = annex_at[k + 1][0] if k + 1 < len(annex_at) else e
        ref = f"{doc['label']}, Annex {num}"
        out.append((f"{doc['id']}.annex{num}", ref, f"{ref}: {_clean(lines[i + 1 : j])}"))
    return out


def parse_basel(text: str, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    ch = doc["chapter"]
    pat = re.compile(rf"^{ch}\.(\d+)$")
    out = []
    for n, plines in _monotonic(text.splitlines(), pat, max_jump=5):
        ref = f"{doc['label']}{ch}.{n}"
        out.append((f"{doc['id']}.{n}", ref, f"{ref}: {_clean(plines)}"))
    return out


MARISK_HEAD = re.compile(r"^((?:AT|BT|BTO|BTR)(?: \d+(?:\.\d+)*)?) ([A-ZÄÖÜ].*)$")


def parse_marisk(text: str, doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    s = _start_index(lines, r"^AT 1 Vorbemerkung$", 2)
    out = []
    section, title = "AT 1", "Vorbemerkung"
    tz, last = 0, 0
    buf: list[str] = []

    def flush() -> None:
        t = _clean(buf)
        if tz and t:
            key = section.lower().replace(" ", "")
            ref = f"MaRisk {section} Tz. {tz}"
            out.append((f"marisk.{key}.tz{tz}", ref, f"{ref} ({title}): {t}"))

    for ln in lines[s + 1 :]:
        st = ln.strip()
        m = MARISK_HEAD.match(st)
        if m and " Tz." not in st and len(st) < 120 and not st.endswith("."):
            flush()
            section, title, tz, buf, last = m.group(1), m.group(2), 0, [], 0
            continue
        if re.fullmatch(r"\d{1,2}", st) and last < int(st) <= last + 2:
            flush()
            tz = last = int(st)
            buf = []
            continue
        buf.append(ln)
    flush()
    return out


PARSERS = {
    "eu_articles": parse_eu_articles,
    "numbered": parse_numbered,
    "basel": parse_basel,
    "marisk": parse_marisk,
}


def build_chunks(check_hash: bool = True) -> list[Chunk]:
    cfg = load_config("corpus")
    d = ROOT / cfg["dir"]
    chunks: list[Chunk] = []
    for doc in cfg["documents"]:
        path = d / doc["file"]
        if check_hash and sha256(path) != doc["sha256"]:
            raise ValueError(f"{doc['id']}: sha256 mismatch for {path}")
        triples = PARSERS[doc["parser"]](pdf_text(path), doc)
        for cid, ref, txt in triples:
            chunks.extend(_split(cid, doc["id"], ref, txt, int(cfg["max_chunk_chars"])))
    return _dedupe(chunks)


def corpus_available() -> bool:
    cfg = load_config("corpus")
    return all((ROOT / cfg["dir"] / doc["file"]).exists() for doc in cfg["documents"])
