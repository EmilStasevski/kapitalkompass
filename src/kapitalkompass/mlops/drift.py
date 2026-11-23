"""Drift monitoring with Evidently: seeded monitoring batches of new loans, some with an injected
real-estate stress, compared with a reference sample of today's book. Writes
results/mlops/drift.json and the HTML report for one strong-shock batch to reports/.

    uv run python -m kapitalkompass.mlops.drift
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from kapitalkompass.config import REPORTS_DIR, SYNTH_DIR, load_config
from kapitalkompass.data.schemas import LoanBook
from kapitalkompass.results import write_json, write_table


def _mean(s: pl.Series) -> float:
    return float(np.mean(s.to_numpy())) if s.len() else 0.0


def apply_shock(
    batch: pl.DataFrame, pool: pl.DataFrame, severity: float, shock: dict[str, Any], seed: int
) -> pl.DataFrame:
    """Real-estate stress of a given severity (0 = none): a shift of the new-business mix
    towards the sector, one-notch downgrades with the grade PD, higher LGD and repricing."""
    if severity <= 0:
        return batch
    rng = np.random.default_rng(seed)
    bank = load_config("bank")["loan_book"]
    grade_pd = {int(g): float(v["pd"]) for g, v in bank["grades"].items()}
    lo, hi = bank["lgd_bounds"]
    sec = shock["sector"]
    n = batch.height
    # mix shift: replace a share of the batch with loans from the shocked sector
    n_extra = round(float(shock["extra_share"]) * severity * n)
    if n_extra:
        sec_pool = pool.filter(pl.col("sector") == sec)
        extra = sec_pool[rng.integers(0, sec_pool.height, n_extra)]
        keep = np.sort(rng.choice(n, n - n_extra, replace=False))
        batch = pl.concat([batch[keep], extra])
    is_sec = (batch["sector"] == sec).to_numpy()
    down = is_sec & (rng.random(batch.height) < float(shock["downgrade_prob"]) * severity)
    grade = np.minimum(batch["grade"].to_numpy() + down, max(grade_pd))
    lgd = batch["lgd"].to_numpy() * np.where(is_sec, 1 + float(shock["lgd_uplift"]) * severity, 1)
    margin = batch["client_margin"].to_numpy() + is_sec * float(shock["margin_add"]) * severity
    return batch.with_columns(
        grade=pl.Series(grade),
        pd=pl.Series([grade_pd[int(g)] for g in grade]),
        lgd=pl.Series(np.clip(lgd, lo, hi)),
        client_margin=pl.Series(margin),
    )


def _dataset(df: pl.DataFrame, cfg: dict[str, Any]) -> Dataset:
    cat = list(cfg["categorical"])
    num = [c for c in cfg["columns"] if c not in cat]
    p: pd.DataFrame = df.select(cfg["columns"]).to_pandas()
    for c in cat:
        p[c] = p[c].astype(str)
    return Dataset.from_pandas(
        p, data_definition=DataDefinition(numerical_columns=num, categorical_columns=cat)
    )


def drift_check(ref: Dataset, cur: Dataset, drift_share: float) -> tuple[dict[str, Any], Any]:
    snap = Report([DataDriftPreset(drift_share=drift_share)]).run(cur, ref)
    out: dict[str, Any] = {"columns": {}}
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            out["drifted_count"] = int(m["value"]["count"])
            out["drifted_share"] = float(m["value"]["share"])
        elif cfg["type"].endswith("ValueDrift"):
            out["columns"][cfg["column"]] = {
                "score": float(m["value"]),
                "method": cfg["method"],
                "threshold": float(cfg["threshold"]),
                "drifted": float(m["value"]) >= float(cfg["threshold"]),
            }
    out["dataset_drift"] = out["drifted_share"] >= drift_share
    return out, snap


def run() -> dict[str, Any]:
    mcfg = load_config("mlops")
    cfg = mcfg["drift"]
    seed = int(mcfg["seed"])
    book = LoanBook.validate(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(book.height)
    n_ref = int(cfg["reference_size"])
    reference = book[perm[:n_ref]]
    pool = book[perm[n_ref:]]  # monitoring batches never reuse reference loans
    ref_ds = _dataset(reference, cfg)

    rows = []
    example = None
    for k, (name, sev) in enumerate(cfg["severities"].items()):
        for b in range(int(cfg["batches_per_severity"])):
            s = seed + 1000 * (k + 1) + b
            idx = np.random.default_rng(s).integers(0, pool.height, int(cfg["batch_size"]))
            batch = apply_shock(pool[idx], pool, float(sev), cfg["shock"], s + 500)
            res, snap = drift_check(ref_ds, _dataset(batch, cfg), float(cfg["drift_share"]))
            rows.append(
                {
                    "severity": name,
                    "severity_value": float(sev),
                    "batch": b,
                    "dataset_drift": res["dataset_drift"],
                    "drifted_count": res["drifted_count"],
                    **{f"score_{c}": v["score"] for c, v in res["columns"].items()},
                    **{f"drift_{c}": v["drifted"] for c, v in res["columns"].items()},
                }
            )
            if name == "strong" and b == 0:
                example = snap
    tab = pl.DataFrame(rows)
    write_table("mlops/drift_batches.csv", tab)
    rate = {
        name: _mean(tab.filter(pl.col("severity") == name)["dataset_drift"])
        for name in cfg["severities"]
    }
    ordered = [rate[n] for n in sorted(cfg["severities"], key=lambda n: cfg["severities"][n])]
    col_rate = {
        name: {
            c: _mean(tab.filter(pl.col("severity") == name)[f"drift_{c}"]) for c in cfg["columns"]
        }
        for name in cfg["severities"]
    }
    drift_cols = [f"drift_{c}" for c in cfg["columns"]]
    any_rule = tab.with_columns(any_drift=pl.any_horizontal(drift_cols))
    out = {
        # Added after the pre-registered H3 result was seen (exploratory, not a test of H3):
        # flag a batch if any single column drifts.
        "exploratory_any_column_rule": {
            name: _mean(any_rule.filter(pl.col("severity") == name)["any_drift"])
            for name in cfg["severities"]
        },
        "batches_per_severity": int(cfg["batches_per_severity"]),
        "batch_size": int(cfg["batch_size"]),
        "reference_size": n_ref,
        "drift_share": float(cfg["drift_share"]),
        "severities": cfg["severities"],
        "detection_rate": rate,
        "detection_monotone": all(a <= b for a, b in pairwise(ordered)),
        "false_alarm_rate": rate["none"],
        "column_drift_rate": col_rate,
        "shocked_batches_flagged": int(
            tab.filter((pl.col("severity_value") > 0) & pl.col("dataset_drift")).height
        ),
        "shocked_batches": int(tab.filter(pl.col("severity_value") > 0).height),
        "method": "Evidently DataDriftPreset: Wasserstein (normed) for numeric, Jensen-Shannon "
        "for categorical and low-cardinality columns, threshold 0.1; dataset drift if the share "
        "of drifted columns >= drift_share",
    }
    write_json("mlops/drift.json", out)
    if example is not None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        example.save_html(str(REPORTS_DIR / "drift_report_strong_shock.html"))
    print(f"drift: detection {rate}, monotone {out['detection_monotone']}")
    return out


if __name__ == "__main__":
    run()
