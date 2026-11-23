"""Dagster job for the whole pipeline (modules 1-5 plus the module-7 drift monitor and the
hypothesis experiments). Every module is an asset that writes results/ and logs one MLflow run;
asset checks re-assert the analytic cross-checks on the written results.

    uv run python -m kapitalkompass.mlops.pipeline         # run the job, fail on a failed check
    uv run dagster dev -m kapitalkompass.mlops.pipeline    # UI
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

import dagster as dg

from kapitalkompass.mlops import tracking
from kapitalkompass.results import read_json

PIPELINE_ID = os.environ.get("KK_PIPELINE_ID") or uuid.uuid4().hex[:12]


def _run(module: str, fn: Callable[[], Any]) -> dg.MaterializeResult:
    t0 = time.perf_counter()
    fn()
    secs = time.perf_counter() - t0
    facts = tracking.log_module(module, secs, PIPELINE_ID)
    return dg.MaterializeResult(
        metadata={
            "seconds": round(secs, 1),
            "mlflow_params": facts["params"],
            "mlflow_metrics": facts["metrics"],
            "results_sha256": facts["results_sha256"][:16],
        }
    )


def _check(ok: bool, **meta: Any) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(passed=bool(ok), metadata={k: str(v) for k, v in meta.items()})


# ----------------------------------------------------------------------------- assets


@dg.asset(group_name="data")
def synthetic_bank() -> dg.MaterializeResult:
    from kapitalkompass.data import synth

    return _run("data", synth.build)


@dg.asset(deps=[synthetic_bank], group_name="modules")
def credit() -> dg.MaterializeResult:
    from kapitalkompass.credit import run

    return _run("credit", run.run)


@dg.asset(deps=[synthetic_bank], group_name="modules")
def bayes() -> dg.MaterializeResult:
    from kapitalkompass.bayes import run

    def both() -> None:
        run.run()
        run.run_calibration()

    return _run("bayes", both)


@dg.asset(deps=[synthetic_bank], group_name="modules")
def alm() -> dg.MaterializeResult:
    from kapitalkompass.alm import run

    return _run("alm", run.run)


@dg.asset(deps=[credit, alm], group_name="modules")
def steering() -> dg.MaterializeResult:
    from kapitalkompass.steering import run

    return _run("steering", run.run)


@dg.asset(deps=[synthetic_bank], group_name="mlops")
def drift() -> dg.MaterializeResult:
    from kapitalkompass.mlops import drift as d

    return _run("drift", d.run)


@dg.asset(deps=[synthetic_bank], group_name="mlops")
def hypothesis_experiments() -> dg.MaterializeResult:
    from kapitalkompass.mlops import experiments

    return _run("experiments", experiments.run)


@dg.asset(deps=[credit, bayes, alm, steering, drift, hypothesis_experiments], group_name="mlops")
def tracking_summary() -> dg.MaterializeResult:
    """Deterministic record of what was logged, plus the hypothesis verdicts."""
    from kapitalkompass.mlops.hypotheses import evaluate
    from kapitalkompass.results import write_json

    modules = ["data", "credit", "bayes", "alm", "steering", "drift", "experiments"]
    facts = {
        m: {
            "params": len(tracking.params_for(m)),
            "metrics": len(tracking.metrics_for(m)),
            "results_sha256": tracking.results_hash(m),
        }
        for m in modules
    }
    write_json(
        "mlops/tracking.json",
        {
            "experiment": "kapitalkompass",
            "runs_per_pipeline": len(modules),
            "modules": facts,
            "data_hash": tracking.data_hash(),
            "tags": ["module", "git_commit", "git_dirty", "data_hash", "results_sha256"],
        },
    )
    verdicts = {h["id"]: h["verdict"] for h in evaluate()["hypotheses"]}
    return dg.MaterializeResult(metadata=verdicts)


# ----------------------------------------------------------------------------- checks


@dg.asset_check(asset=synthetic_bank, description="Generator is deterministic (content hashes)")
def synth_hashes_stable() -> dg.AssetCheckResult:
    from kapitalkompass.config import SYNTH_DIR
    from kapitalkompass.data import synth

    again = synth.build(write=False)
    stored: dict[str, str] = {}
    for line in (SYNTH_DIR / "HASHES").read_text().splitlines():
        digest, name = line.split("  ")
        stored[name] = digest
    same = all(synth.frame_hash(df) == stored[name] for name, df in again.items())
    return _check(same, tables=len(again))


@dg.asset_check(asset=credit, description="Euler ES contributions sum to total ES")
def euler_additivity() -> dg.AssetCheckResult:
    s = read_json("credit/summary.json")
    gap = abs(s["euler_es_sum_eur"] / s["es_eur"] - 1)
    return _check(gap < 1e-6, relative_gap=gap)


@dg.asset_check(asset=credit, description="One-factor MC UL close to ASRF UL (granular book)")
def asrf_cross_check() -> dg.AssetCheckResult:
    r = read_json("credit/summary.json")["one_factor"]["ul_over_asrf_ul"]
    return _check(0.97 <= r <= 1.10, ul_over_asrf_ul=r)


@dg.asset_check(asset=credit, description="Simulated EL matches analytic EL within 1%")
def el_cross_check() -> dg.AssetCheckResult:
    s = read_json("credit/summary.json")
    gap = abs(s["el_simulated_eur"] / s["el_analytic_eur"] - 1)
    return _check(gap < 0.01, relative_gap=gap)


@dg.asset_check(asset=bayes, description="No divergences; truth recovery near nominal")
def bayes_recovery() -> dg.AssetCheckResult:
    s, c = read_json("bayes/summary.json"), read_json("bayes/calibration.json")
    cov90 = c["coverage"]["0.9"]
    return _check(
        s["mcmc"]["divergences"] == 0 and 0.8 <= cov90 <= 0.98, divergences=0, coverage_90=cov90
    )


@dg.asset_check(asset=alm, description="Zero shock gives zero ΔEVE and ΔNII; LCR, NSFR ≥ 100%")
def alm_identities() -> dg.AssetCheckResult:
    s = read_json("alm/summary.json")
    none = s["irrbb"]["scenarios"]["none"]
    ok = none["delta_eve"] == 0 and none["delta_nii"] == 0
    return _check(ok and s["lcr"]["lcr"] >= 1 and s["nsfr"]["nsfr"] >= 1, lcr=s["lcr"]["lcr"])


@dg.asset_check(asset=steering, description="Allocation additive, optimiser feasible")
def steering_consistency() -> dg.AssetCheckResult:
    s = read_json("steering/summary.json")
    ok = s["euler_allocation_sums_to_total"] and not s["optimiser"]["feasibility_violations"]
    return _check(ok, raroc_after=s["optimiser"]["raroc_after_resimulated"])


@dg.asset_check(asset=drift, description="No false alarms on unshocked batches")
def drift_no_false_alarms() -> dg.AssetCheckResult:
    d = read_json("mlops/drift.json")
    return _check(d["false_alarm_rate"] <= 0.1, false_alarm_rate=d["false_alarm_rate"])


@dg.asset_check(
    asset=drift,
    description="Injected strong shock is visible at column level (any-column rule)",
    blocking=False,
)
def drift_shock_visible() -> dg.AssetCheckResult:
    d = read_json("mlops/drift.json")
    r = d["exploratory_any_column_rule"]["strong"]
    return dg.AssetCheckResult(
        passed=r >= 0.9,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={
            "any_column_rule_strong": str(r),
            "dataset_rule_strong": str(d["detection_rate"]["strong"]),
        },
    )


ASSETS = [
    synthetic_bank,
    credit,
    bayes,
    alm,
    steering,
    drift,
    hypothesis_experiments,
    tracking_summary,
]
CHECKS = [
    synth_hashes_stable,
    euler_additivity,
    asrf_cross_check,
    el_cross_check,
    bayes_recovery,
    alm_identities,
    steering_consistency,
    drift_no_false_alarms,
    drift_shock_visible,
]
pipeline_job = dg.define_asset_job("pipeline", selection=dg.AssetSelection.all())
defs = dg.Definitions(assets=ASSETS, asset_checks=CHECKS, jobs=[pipeline_job])


def main() -> None:
    os.environ["KK_PIPELINE_ID"] = PIPELINE_ID
    job = defs.resolve_job_def("pipeline")
    result = job.execute_in_process(instance=dg.DagsterInstance.ephemeral())
    failed = []
    for ev in result.get_asset_check_evaluations():
        sev = ev.severity.value if ev.severity else "ERROR"
        print(
            f"check {ev.asset_key.to_user_string()}.{ev.check_name}: "
            f"{'pass' if ev.passed else 'FAIL (' + sev + ')'}"
        )
        if not ev.passed and sev == "ERROR":
            failed.append(ev.check_name)
    if not result.success or failed:
        sys.exit(f"pipeline failed: {failed or 'step error'}")
    print(f"pipeline ok (id {PIPELINE_ID})")


if __name__ == "__main__":
    main()
