"""MLflow tracking: one run per pipeline module with its config as params, every numeric leaf of
its results JSON as metrics, and git commit, data hash and results hash as tags.

The store is local (sqlite in mlruns/, git-ignored). results/mlops/tracking.json records only
deterministic facts (counts and hashes); run ids and timestamps stay in MLflow.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kapitalkompass.config import (
    CONFIG_DIR,
    EXTERNAL_DIR,
    RESULTS_DIR,
    ROOT,
    SYNTH_DIR,
    load_config,
)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# module -> (config files logged as params, results JSON files logged as metrics)
MODULES: dict[str, tuple[list[str], list[str]]] = {
    "data": (["bank"], []),
    "credit": (["model"], ["credit/summary.json"]),
    "bayes": (["model"], ["bayes/summary.json", "bayes/calibration.json"]),
    "alm": (["alm"], ["alm/summary.json"]),
    "steering": (["steering"], ["steering/summary.json"]),
    "drift": (["mlops"], ["mlops/drift.json"]),
    "experiments": (["mlops"], ["mlops/hypotheses_experiments.json"]),
    "agent": (["agent"], ["agent/eval_summary.json"]),
}
RESULT_DIRS = {
    "data": [],
    "credit": ["credit"],
    "bayes": ["bayes"],
    "alm": ["alm"],
    "steering": ["steering"],
    "drift": ["mlops/drift.json", "mlops/drift_batches.csv"],
    "experiments": ["mlops/hypotheses_experiments.json", "mlops/granularity.csv"],
    "agent": ["agent/eval_summary.json", "agent/eval_details.csv"],
}
EXCLUDE = {"credit/benchmark.json"}  # hardware timings, not reproducible (decision 15)


def git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def data_hash() -> str:
    """Hash of the synthetic-bank content hashes plus the rate cache and all config files."""
    h = hashlib.sha256()
    h.update((SYNTH_DIR / "HASHES").read_bytes())
    h.update((EXTERNAL_DIR / "ecb_yc_spot.parquet").read_bytes())
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def _files(rels: list[str]) -> Iterator[Path]:
    for rel in rels:
        p = RESULTS_DIR / rel
        if p.is_dir():
            yield from sorted(
                q
                for q in p.rglob("*")
                if q.is_file() and str(q.relative_to(RESULTS_DIR)) not in EXCLUDE
            )
        elif p.exists():
            yield p


def results_hash(module: str) -> str:
    h = hashlib.sha256()
    for p in _files(RESULT_DIRS[module]):
        h.update(str(p.relative_to(RESULTS_DIR)).encode() + p.read_bytes())
    return h.hexdigest()


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list) and d and all(isinstance(x, int | float) for x in d):
        for i, v in enumerate(d):
            out[f"{prefix}{i}"] = v
    else:
        out[prefix.rstrip(".")] = d
    return out


def _key(k: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", k)[:250]


def params_for(module: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in MODULES[module][0]:
        for k, v in flatten(load_config(name)).items():
            out[_key(f"{name}.{k}")] = str(v)[:6000]
    return out


def metrics_for(module: str) -> dict[str, float]:
    import json

    out: dict[str, float] = {}
    for rel in MODULES[module][1]:
        p = RESULTS_DIR / rel
        if not p.exists():
            continue
        stem = rel.split("/")[-1].removesuffix(".json")
        for k, v in flatten(json.loads(p.read_text())).items():
            finite = isinstance(v, int | float) and v == v and abs(v) != float("inf")
            if isinstance(v, bool) or finite:
                out[_key(f"{stem}.{k}")] = float(v)
    return out


def _setup() -> Any:
    import mlflow

    cfg = load_config("mlops")["mlflow"]
    (ROOT / "mlruns").mkdir(exist_ok=True)
    uri = cfg["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(
            cfg["experiment"], artifact_location=(ROOT / cfg["artifact_root"]).as_uri()
        )
    mlflow.set_experiment(cfg["experiment"])
    return mlflow


def log_module(module: str, seconds: float | None = None, pipeline_id: str = "") -> dict[str, Any]:
    """Log one finished module run to MLflow; returns the deterministic facts about it."""
    mlflow = _setup()
    sha, dirty = git_commit()
    params, metrics = params_for(module), metrics_for(module)
    facts = {
        "params": len(params),
        "metrics": len(metrics),
        "results_sha256": results_hash(module),
    }
    with mlflow.start_run(run_name=module):
        mlflow.set_tags(
            {
                "module": module,
                "git_commit": sha,
                "git_dirty": str(dirty),
                "data_hash": data_hash(),
                "results_sha256": facts["results_sha256"],
                "pipeline_id": pipeline_id,
            }
        )
        items = list(params.items())
        for i in range(0, len(items), 90):
            mlflow.log_params(dict(items[i : i + 90]))
        if metrics:
            mlflow.log_metrics(metrics)
        if seconds is not None:
            mlflow.log_metric("runtime_seconds", seconds)
        for p in _files(RESULT_DIRS[module]):
            if p.stat().st_size < 5_000_000:
                mlflow.log_artifact(str(p), artifact_path=str(p.parent.relative_to(RESULTS_DIR)))
    return facts


def compare_last_runs() -> dict[str, Any]:
    """Reproducibility check from the MLflow history: for every module, compare the metrics and
    results hash of its last two runs with the same data hash. Written to
    results/mlops/reproducibility.json, which depends on the run history and is therefore
    outside the bit-for-bit claim (like the benchmark)."""
    from kapitalkompass.results import write_json

    mlflow = _setup()
    dh = data_hash()
    out: dict[str, Any] = {"data_hash": dh[:16], "modules": {}}
    for module in MODULES:
        runs = mlflow.search_runs(
            filter_string=f"tags.module = '{module}' and tags.data_hash = '{dh}'",
            order_by=["start_time DESC"],
            max_results=2,
            output_format="list",
        )
        if len(runs) < 2:
            out["modules"][module] = {"compared": False, "runs_with_this_data": len(runs)}
            continue
        a, b = runs[0].data, runs[1].data
        keys = sorted((set(a.metrics) & set(b.metrics)) - {"runtime_seconds"}, key=str)
        rel = [
            abs(a.metrics[k] - b.metrics[k]) / max(abs(a.metrics[k]), abs(b.metrics[k]), 1e-12)
            for k in keys
        ]
        out["modules"][module] = {
            "compared": True,
            "metrics_compared": len(keys),
            "identical_metrics": sum(r == 0 for r in rel),
            "max_rel_diff": max(rel) if rel else 0.0,
            "results_hash_equal": a.tags.get("results_sha256") == b.tags.get("results_sha256"),
        }
    write_json("mlops/reproducibility.json", out)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(compare_last_runs(), indent=2))
