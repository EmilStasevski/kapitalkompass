"""Everything the reports need, read from results/ only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from kapitalkompass.config import RESULTS_DIR
from kapitalkompass.results import read_json, read_table


def _opt_json(rel: str) -> dict[str, Any] | None:
    return read_json(rel) if (RESULTS_DIR / rel).exists() else None


@dataclass
class Facts:
    credit: dict[str, Any]
    bench: dict[str, Any] | None
    seg_capital: pl.DataFrame
    bayes: dict[str, Any]
    calib: dict[str, Any]
    seg_pd: pl.DataFrame
    alm: dict[str, Any]
    irrbb: pl.DataFrame
    steer: dict[str, Any]
    raroc: pl.DataFrame
    alloc: pl.DataFrame
    actions: dict[str, Any]
    limits_before: pl.DataFrame
    limits_after: pl.DataFrame
    limits_actions: pl.DataFrame
    agent: dict[str, Any] | None
    heldout: dict[str, Any] | None
    drift: dict[str, Any] | None
    experiments: dict[str, Any] | None
    tracking: dict[str, Any] | None
    repro: dict[str, Any] | None
    hypotheses: dict[str, Any] | None

    @property
    def cuda_speedup(self) -> float | None:
        if not self.bench:
            return None
        runs = {r["engine"]: r for r in self.bench["runs"]}
        return runs["torch_cuda"]["speedup_vs_numpy"] if "torch_cuda" in runs else None

    @property
    def numba_speedup(self) -> float | None:
        if not self.bench:
            return None
        runs = {r["engine"]: r for r in self.bench["runs"]}
        if "torch_cuda" not in runs or "numba_parallel" not in runs:
            return None
        return float(
            runs["torch_cuda"]["speedup_vs_numpy"] / runs["numba_parallel"]["speedup_vs_numpy"]
        )

    @property
    def energy_ratio(self) -> float | None:
        """Energy per million scenarios, numpy (1 core) over CUDA."""
        if not self.bench:
            return None
        runs = {r["engine"]: r for r in self.bench["runs"]}
        if "torch_cuda" not in runs or "numpy_1core" not in runs:
            return None
        return float(
            runs["numpy_1core"]["joules_per_million_scenarios"]
            / runs["torch_cuda"]["joules_per_million_scenarios"]
        )

    @property
    def repro_identical(self) -> tuple[int, int]:
        """(identical, compared) metrics over all modules of the last two pipeline runs."""
        if not self.repro:
            return 0, 0
        mods = [m for m in self.repro["modules"].values() if m.get("compared")]
        return (
            sum(int(m["identical_metrics"]) for m in mods),
            sum(int(m["metrics_compared"]) for m in mods),
        )

    def limit_counts(self, df: pl.DataFrame) -> dict[str, int]:
        return {s: int((df["status"] == s).sum()) for s in ("green", "amber", "red")}


def load() -> Facts:
    return Facts(
        credit=read_json("credit/summary.json"),
        bench=_opt_json("credit/benchmark.json"),
        seg_capital=read_table("credit/segment_capital.csv"),
        bayes=read_json("bayes/summary.json"),
        calib=read_json("bayes/calibration.json"),
        seg_pd=read_table("bayes/segment_pd.csv"),
        alm=read_json("alm/summary.json"),
        irrbb=read_table("alm/irrbb_scenarios.csv"),
        steer=read_json("steering/summary.json"),
        raroc=read_table("steering/raroc.csv"),
        alloc=read_table("steering/allocation.csv"),
        actions=read_json("steering/actions.json"),
        limits_before=read_table("steering/limits_before.csv"),
        limits_after=read_table("steering/limits_after.csv"),
        limits_actions=read_table("steering/limits_after_actions.csv"),
        agent=_opt_json("agent/eval_summary.json"),
        heldout=_opt_json("agent/heldout_summary.json"),
        drift=_opt_json("mlops/drift.json"),
        experiments=_opt_json("mlops/hypotheses_experiments.json"),
        tracking=_opt_json("mlops/tracking.json"),
        repro=_opt_json("mlops/reproducibility.json"),
        hypotheses=_opt_json("mlops/hypotheses.json"),
    )
