"""LanceDB vector index over the regulatory corpus (bge-m3 embeddings) plus exact-reference lookup.

The chunk-id manifest ``data/corpus_chunks.csv`` (ids, refs, text hashes; no text) is committed,
so a test can check that re-chunking the pinned PDFs reproduces exactly the same ids.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any

import lancedb
import polars as pl

from kapitalkompass.agent import ollama
from kapitalkompass.agent.corpus import Chunk, build_chunks
from kapitalkompass.config import ROOT, load_config

MANIFEST = ROOT / "data" / "corpus_chunks.csv"


def manifest(chunks: list[Chunk]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [c.id for c in chunks],
            "doc": [c.doc for c in chunks],
            "ref": [c.ref for c in chunks],
            "n_chars": [len(c.text) for c in chunks],
            "text_sha256": [hashlib.sha256(c.text.encode()).hexdigest()[:16] for c in chunks],
        }
    )


def build_index() -> dict[str, Any]:
    cfg = load_config("agent")["index"]
    chunks = build_chunks()
    vecs = ollama.embed([c.text for c in chunks])
    db = lancedb.connect(str(ROOT / cfg["dir"]))
    data = [
        {"id": c.id, "doc": c.doc, "ref": c.ref, "text": c.text, "vector": v}
        for c, v in zip(chunks, vecs, strict=True)
    ]
    db.create_table(cfg["table"], data=data, mode="overwrite")
    manifest(chunks).write_csv(MANIFEST)
    _table.cache_clear()
    return {"chunks": len(chunks), "dim": len(vecs[0])}


@lru_cache(maxsize=1)
def _table() -> Any:
    cfg = load_config("agent")["index"]
    return lancedb.connect(str(ROOT / cfg["dir"])).open_table(cfg["table"])


@lru_cache(maxsize=1)
def _all_rows() -> list[dict[str, Any]]:
    return _table().to_arrow().select(["id", "doc", "ref", "text"]).to_pylist()


REF_PATTERNS = [
    # "Art. 395 CRR", "Artikel 92", "Article 428b"
    (r"\b(?:Art\.?|Artikel|Article)\s*(\d+[a-z]*)", lambda m: rf"^Art\. {m.group(1)}\b"),
    # "BTR 3.1", "AT 4.3.2" (optionally "Tz. 4")
    (
        r"\b((?:AT|BTO|BTR|BT)\s*\d+(?:\.\d+)*)(?:\s*Tz\.?\s*(\d+))?",
        lambda m: (
            rf"^MaRisk {re.escape(m.group(1))}"
            + (rf" Tz\. {m.group(2)}$" if m.group(2) else r" Tz\.")
        ),
    ),
    # "LCR30.40", "NSF30.10", "CRE31.5"
    (r"\b((?:LCR|NSF|CRE)\d{2}\.\d+)\b", lambda m: rf"^{re.escape(m.group(1))}$"),
]


def exact_refs(query: str, limit: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pat, to_ref in REF_PATTERNS:
        for m in re.finditer(pat, query):
            rx = re.compile(to_ref(m))
            hits.extend(r for r in _all_rows() if rx.search(r["ref"]))
    seen: set[str] = set()
    out = []
    for h in hits:
        if h["id"] not in seen:
            seen.add(h["id"])
            out.append(h)
    return out[:limit]


def search(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Exact references named in the query first, then nearest neighbours (cosine)."""
    cfg = load_config("agent")["index"]
    k = int(k or cfg["top_k"])
    exact = exact_refs(query, k)
    qv = ollama.embed([query])[0]
    res = _table().search(qv).metric("cosine").limit(k + len(exact)).to_list()
    out = [{**e, "score": 1.0} for e in exact]
    ids = {e["id"] for e in exact}
    for r in res:
        if r["id"] not in ids and len(out) < k:
            out.append(
                {
                    "id": r["id"],
                    "doc": r["doc"],
                    "ref": r["ref"],
                    "text": r["text"],
                    "score": 1.0 - float(r["_distance"]),
                }
            )
    return out
