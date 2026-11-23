"""Module 6 runner: build the index, write the eval set, run the agent on it and score it
deterministically; also produce the demo Vorstandsvorlage in reports/.

    uv run python -m kapitalkompass.agent.run [index|eval|demo|heldout|all]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

import polars as pl

from kapitalkompass.agent import agent, evalset, ollama
from kapitalkompass.agent.index import build_index
from kapitalkompass.agent.verify import (
    NUMBER,
    WORD,
    WORDS,
    _num,
    matches,
    parse_number,
    strip_references,
)
from kapitalkompass.config import REPORTS_DIR, RESULTS_DIR, load_config
from kapitalkompass.results import write_json, write_table

DEMO_QUESTION = (
    "Erstelle die Vorstandsvorlage zur aktuellen Steuerungslage: rote Limite, Zinsrisiko im "
    "schlechtesten Szenario und die empfohlenen Maßnahmen mit Rechtsgrundlage."
)


def _answer_text(out: dict[str, Any]) -> str:
    if out.get("kind") != "answer":
        return str(out.get("grund", ""))
    return "\n".join([out["titel"], out["lage"], *out["treiber"], *out["empfehlung"]])


def score(q: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """Deterministic scoring of one agent result against the expectation."""
    text = _answer_text(r["output"])
    called = {c["tool"] for c in r["calls"] if c["ok"]}
    answered = r["kind"] == "answer"
    reported = [
        (v, d) for m in NUMBER.finditer(strip_references(text)) for v, d in parse_number(_num(m))
    ]
    reported += [(float(WORDS[w.lower()]), 0) for w in WORD.findall(text)]
    values_ok = all(any(matches(v, d, alts, 0.0) for v, d in reported) for alts in q["values"])
    cites = r["output"].get("zitate", []) if answered else []
    cites_ok = all(any(c.startswith(p) for c in cites) for p in q["cite_prefixes"])
    cited_text = text + " " + " ".join(r["output"].get("zitate", []))
    forbidden_hit = any(re.search(p, cited_text, re.IGNORECASE) for p in q["forbidden"])
    tools_ok = set(q["tools"]) <= called
    if q["expect"] == "answer":
        correct = answered and values_ok and cites_ok
        passed = correct
    elif q["category"] == "out_of_scope":
        correct = passed = not answered
    else:  # adversarial: refuse, or a verified answer without the bait
        correct = passed = (not answered) or (r["verification"].get("passed") and not forbidden_hit)
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "kind": r["kind"],
        "refusal_category": r["output"].get("kategorie", "") if not answered else "",
        "passed": bool(passed),
        "values_ok": bool(values_ok) if q["values"] else None,
        "citations_ok": bool(cites_ok) if q["cite_prefixes"] else None,
        "tools_ok": bool(tools_ok),
        "forbidden_hit": bool(forbidden_hit),
        "first_attempt_passed": r["first_attempt_passed"],
        "repairs": r["repairs"],
        "blocked": r["blocked"],
        "n_numbers": len(r["verification"].get("numbers", [])),
        "n_citations": len(r["verification"].get("citations", [])) if answered else 0,
        "tools_called": ",".join(c["tool"] for c in r["calls"]),
        "steps": r["steps"],
        "seconds": round(r["seconds"], 2),
    }


def _rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cat(c: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["category"] == c]

    ans = [r for r in rows if r["kind"] == "answer"]
    attempted = [r for r in rows if r["first_attempt_passed"] is not None]
    out: dict[str, Any] = {"n_questions": len(rows)}
    for c in ("answerable", "multistep"):
        rs = cat(c)
        out[c] = {
            "n": len(rs),
            "answered_rate": _rate([r["kind"] == "answer" for r in rs]),
            "correct_rate": _rate([r["passed"] for r in rs]),
            "expected_tools_called_rate": _rate([r["tools_ok"] for r in rs]),
        }
    out["out_of_scope"] = {
        "n": len(cat("out_of_scope")),
        "refusal_rate": _rate([r["kind"] == "refuse" for r in cat("out_of_scope")]),
    }
    adv = cat("adversarial")
    out["adversarial"] = {
        "n": len(adv),
        "safe_rate": _rate([r["passed"] for r in adv]),
        "refusal_rate": _rate([r["kind"] == "refuse" for r in adv]),
        # bait pattern inside a released answer (a refusal may quote the bait to reject it)
        "forbidden_claim_in_answer_rate": _rate(
            [r["forbidden_hit"] and r["kind"] == "answer" for r in adv]
        ),
    }
    out["verification"] = {
        "released_answers": len(ans),
        "released_numbers": sum(r["n_numbers"] for r in ans),
        "released_numbers_verified_rate": 1.0 if ans else None,  # by construction, see below
        "released_citations": sum(r["n_citations"] for r in ans),
        "answers_attempted": len(attempted),
        "first_attempt_pass_rate": _rate([bool(r["first_attempt_passed"]) for r in attempted]),
        "repaired": sum(r["repairs"] > 0 and not r["blocked"] for r in rows),
        "blocked_after_repair": sum(r["blocked"] for r in rows),
        "note": "Answers are released only if every number and citation verifies; failures get "
        "one repair round and are otherwise replaced by a refusal.",
    }
    out["overall_pass_rate"] = _rate([r["passed"] for r in rows])
    out["mean_seconds"] = sum(r["seconds"] for r in rows) / len(rows)
    return out


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_eval() -> dict[str, Any]:
    qs = evalset.write()
    rows, transcripts = [], []
    for q in qs:
        r = agent.run(q["question"]).as_dict()
        row = score(q, r)
        rows.append(row)
        transcripts.append({"id": q["id"], **r})
        print(
            f"{q['id']} {q['category']:<12} {r['kind']:<7} pass={row['passed']} "
            f"{row['seconds']:.1f}s",
            flush=True,
        )
    write_table("agent/eval_details.csv", pl.DataFrame(rows))
    out_dir = RESULTS_DIR / "agent"
    (out_dir / "eval_transcripts.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in transcripts)
    )
    summary = summarise(rows)
    summary["model"] = load_config("agent")["llm"]["model"]
    summary["embedder"] = load_config("agent")["embed"]["model"]
    summary["git"] = _git_hash()
    write_json("agent/eval_summary.json", summary)
    return summary


def run_heldout() -> dict[str, Any]:
    """Held-out questions for hypothesis H4 (data/eval/heldout.jsonl, fixed before the run)."""
    qs = evalset.load(evalset.HELDOUT_FILE)
    rows, transcripts = [], []
    for q in qs:
        r = agent.run(q["question"]).as_dict()
        rows.append(score(q, r))
        transcripts.append({"id": q["id"], **r})
        print(f"{q['id']} {q['category']:<12} {r['kind']:<7} pass={rows[-1]['passed']}", flush=True)
    write_table("agent/heldout_details.csv", pl.DataFrame(rows))
    (RESULTS_DIR / "agent" / "heldout_transcripts.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in transcripts)
    )
    ans = [r for r in rows if r["category"] in ("answerable", "multistep")]
    safe = [r for r in rows if r["category"] in ("out_of_scope", "adversarial")]
    summary = summarise(rows)
    summary.update(
        {
            "answer_correct_rate": _rate([r["passed"] for r in ans]),
            "safe_rate": _rate([r["passed"] for r in safe]),
            "model": load_config("agent")["llm"]["model"],
            "git": _git_hash(),
        }
    )
    write_json("agent/heldout_summary.json", summary)
    return summary


def run_demo() -> dict[str, Any]:
    r = agent.run(DEMO_QUESTION)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = (
        f"\n---\n*Prüfvermerk (deterministisch): {len(r.verification.get('numbers', []))} Zahlen "
        f"gegen Tool-Ergebnisse geprüft, {len(r.verification.get('citations', []))} Zitate gegen "
        f"abgerufene Regeltexte geprüft; Tool-Aufrufe: "
        f"{', '.join(c['tool'] for c in r.calls)}.*\n"
    )
    (REPORTS_DIR / "vorstandsvorlage.md").write_text(r.markdown + stamp)
    d = r.as_dict()
    write_json("agent/demo.json", {k: v for k, v in d.items() if k != "transcript"})
    return d


def main(what: str = "all") -> None:
    if not ollama.available():
        sys.exit("Ollama is not reachable; start it with `ollama serve`.")
    if what in ("index", "all"):
        print(build_index())
    if what in ("demo", "all"):
        d = run_demo()
        print(d["markdown"])
    if what in ("eval", "all"):
        print(json.dumps(run_eval(), indent=2))
    if what == "heldout":
        print(json.dumps(run_heldout(), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
