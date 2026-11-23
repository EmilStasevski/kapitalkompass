"""Evaluate the decision rules in config/hypotheses.yaml against results/ (no hand-typed
verdicts). Writes results/mlops/hypotheses.json; docs/hypotheses.md is rendered from it."""

from __future__ import annotations

import json
import operator
from typing import Any

from kapitalkompass.config import RESULTS_DIR, load_config
from kapitalkompass.results import write_json

OPS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
}
SYMBOL = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥", "eq": "="}


class MissingResultError(Exception):
    pass


def resolve(ref: str) -> Any:
    rel, key = ref.split(":", 1)
    p = RESULTS_DIR / rel
    if not p.exists():
        raise MissingResultError(rel)
    v: Any = json.loads(p.read_text())
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            raise MissingResultError(ref)
        v = v[part]
    return v


def _value(x: Any) -> Any:
    return resolve(x) if isinstance(x, str) and ":" in x else x


def evaluate_one(h: dict[str, Any], kind: str) -> dict[str, Any]:
    conds = []
    verdict = "supported"
    for c in h["rule"]:
        try:
            left = resolve(c["path"])
            if "minus" in c:
                left = left - resolve(c["minus"])
            right = _value(c["value"])
            ok = bool(OPS[c["op"]](left, right)) if left is not None else False
        except MissingResultError as e:
            conds.append({"path": c["path"], "missing": str(e)})
            verdict = "not run"
            continue
        conds.append(
            {
                "path": c["path"] + (f" − {c['minus']}" if "minus" in c else ""),
                "op": SYMBOL[c["op"]],
                "left": left,
                "right": right,
                "holds": ok,
            }
        )
        if not ok and verdict == "supported":
            verdict = "not supported"
    return {
        "id": h["id"],
        "kind": kind,
        "title": h["title"],
        "statement": " ".join(h["statement"].split()),
        "experiment": " ".join(h.get("experiment", "").split()),
        "conditions": conds,
        "verdict": verdict,
    }


def evaluate() -> dict[str, Any]:
    cfg = load_config("hypotheses")
    rows = [evaluate_one(h, "preregistered") for h in cfg["preregistered"]]
    rows += [evaluate_one(h, "post_hoc") for h in cfg["post_hoc"]]
    out = {"hypotheses": rows}
    write_json("mlops/hypotheses.json", out)
    return out


if __name__ == "__main__":
    for r in evaluate()["hypotheses"]:
        print(r["id"], r["verdict"], r["title"])
