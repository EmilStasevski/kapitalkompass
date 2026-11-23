"""Minimal Ollama HTTP client (chat with JSON-schema constrained decoding, embeddings)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from kapitalkompass.config import load_config


def _cfg() -> dict[str, Any]:
    return load_config("agent")


def available(timeout: float = 2.0) -> bool:
    try:
        httpx.get(f"{_cfg()['llm']['host']}/api/version", timeout=timeout).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def chat(messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
    """One constrained chat completion. Returns the parsed JSON object and token counts."""
    c = _cfg()["llm"]
    r = httpx.post(
        f"{c['host']}/api/chat",
        json={
            "model": c["model"],
            "messages": messages,
            "stream": False,
            "think": bool(c["think"]),
            "format": schema,
            "keep_alive": c["keep_alive"],
            "options": {
                "temperature": float(c["temperature"]),
                "seed": int(c["seed"]),
                "num_ctx": int(c["num_ctx"]),
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    return {
        "content": json.loads(body["message"]["content"]),
        "prompt_tokens": int(body.get("prompt_eval_count", 0)),
        "output_tokens": int(body.get("eval_count", 0)),
    }


def _embed_call(texts: list[str], keep_alive: Any) -> httpx.Response:
    c = _cfg()
    return httpx.post(
        f"{c['llm']['host']}/api/embed",
        json={"model": c["embed"]["model"], "input": texts, "keep_alive": keep_alive},
        timeout=600,
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Batched embeddings. Ollama 0.13.5 returns NaN (HTTP 500) for a few short inputs with
    bge-m3 (e.g. LCR30.38); such a text is re-embedded alone, then with trailing words dropped
    one at a time until the server returns a finite vector (deterministic fallback)."""
    c = _cfg()
    out: list[list[float]] = []
    bs = int(c["embed"]["batch"])
    keep = c["embed"]["keep_alive"]
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        last = i + bs >= len(texts)
        r = _embed_call(batch, keep if last else "1m")
        if r.status_code == 200:
            out.extend(r.json()["embeddings"])
            continue
        for t in batch:
            words = t.split()
            variants = [t] + [" ".join(words[:-k]) for k in range(1, min(12, len(words)))]
            for v in variants:
                r = _embed_call([v], "1m")
                if r.status_code == 200:
                    break
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        if last:
            _embed_call(["."], keep)
    return out
