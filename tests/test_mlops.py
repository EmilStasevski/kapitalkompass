"""Module 7 and reporting: drift monitor, hypothesis rules, tracking helpers, generated docs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from kapitalkompass.config import RESULTS_DIR, ROOT, SYNTH_DIR, load_config
from kapitalkompass.mlops import hypotheses as hyp
from kapitalkompass.mlops.tracking import flatten
from kapitalkompass.report.fmt import Fmt
from kapitalkompass.report.pdf import esc

HAS_BOOK = (SYNTH_DIR / "loan_book.parquet").exists()


@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    if not HAS_BOOK:
        pytest.skip("synthetic bank not generated")
    return pl.read_parquet(SYNTH_DIR / "loan_book.parquet")


def test_shock_zero_is_identity_and_full_shock_moves_the_sector(book: pl.DataFrame) -> None:
    from kapitalkompass.mlops.drift import apply_shock

    shock = load_config("mlops")["drift"]["shock"]
    batch = book[:4000]
    assert apply_shock(batch, book, 0.0, shock, 1).equals(batch)
    out = apply_shock(batch, book, 1.0, shock, 1)
    assert out.height == batch.height

    def share(df: pl.DataFrame) -> float:
        return float((df["sector"] == "real_estate").mean())  # type: ignore[arg-type]

    assert share(out) > share(batch) + 0.05
    re_before = batch.filter(pl.col("sector") == "real_estate")
    re_after = out.filter(pl.col("sector") == "real_estate")
    assert re_after["lgd"].mean() > re_before["lgd"].mean()  # type: ignore[operator]
    assert re_after["grade"].mean() > re_before["grade"].mean()  # type: ignore[operator]
    # other sectors keep their credit parameters
    other = out.filter(pl.col("sector") != "real_estate")
    assert other["lgd"].max() <= 0.95 and other["grade"].max() <= 4  # type: ignore[operator]
    grade_pd = {int(g): v["pd"] for g, v in load_config("bank")["loan_book"]["grades"].items()}
    assert all(grade_pd[g] == p for g, p in zip(out["grade"], out["pd"], strict=True))


def test_drift_check_flags_a_gross_shift_and_not_a_resample(book: pl.DataFrame) -> None:
    from kapitalkompass.mlops.drift import _dataset, drift_check

    cfg = load_config("mlops")["drift"]
    ref = _dataset(book[:3000], cfg)
    same, _ = drift_check(ref, _dataset(book[3000:6000], cfg), 0.3)
    assert not same["dataset_drift"]
    shifted = book[3000:6000].with_columns(
        lgd=pl.col("lgd") * 2, client_margin=pl.col("client_margin") + 0.02, pd=pl.col("pd") * 3
    )
    moved, _ = drift_check(ref, _dataset(shifted, cfg), 0.3)
    assert moved["dataset_drift"] and moved["columns"]["lgd"]["drifted"]


def test_hypothesis_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "r.json").write_text(json.dumps({"a": {"b": 0.3}, "c": 0.1}))
    monkeypatch.setattr(hyp, "RESULTS_DIR", tmp_path)
    ok = {
        "id": "T",
        "title": "t",
        "statement": "s",
        "rule": [
            {"path": "x/r.json:a.b", "op": "gt", "value": 0.2},
            {"path": "x/r.json:a.b", "minus": "x/r.json:c", "op": "ge", "value": "x/r.json:c"},
        ],
    }
    assert hyp.evaluate_one(ok, "post_hoc")["verdict"] == "supported"
    bad = {**ok, "rule": [{"path": "x/r.json:c", "op": "ge", "value": 0.5}]}
    assert hyp.evaluate_one(bad, "post_hoc")["verdict"] == "not supported"
    missing = {**ok, "rule": [{"path": "y/none.json:c", "op": "ge", "value": 0.5}]}
    assert hyp.evaluate_one(missing, "post_hoc")["verdict"] == "not run"


def test_hypothesis_config_paths_are_well_formed() -> None:
    cfg = load_config("hypotheses")
    ids = [h["id"] for h in cfg["preregistered"] + cfg["post_hoc"]]
    assert len(ids) == len(set(ids))
    for h in cfg["preregistered"] + cfg["post_hoc"]:
        for c in h["rule"]:
            assert c["op"] in hyp.OPS and ":" in c["path"]


def test_flatten_and_formatting() -> None:
    assert flatten({"a": {"b": 1, "c": [1.0, 2.0]}, "d": "x"}) == {
        "a.b": 1,
        "a.c.0": 1.0,
        "a.c.1": 2.0,
        "d": "x",
    }
    de, en = Fmt("de"), Fmt("en")
    assert de.num(1234567.891, 2) == "1.234.567,89"
    assert en.num(1234567.891, 2) == "1,234,567.89"
    assert de.pct(-0.2134) == "−21,3 %" and en.pct(0.159) == "15.9%"
    assert de.eur_m(203854557.5) == "203,9 Mio. EUR"
    assert esc("a_b #c $d") == "a\\_b \\#c \\$d"


@pytest.mark.skipif(
    not (RESULTS_DIR / "steering" / "summary.json").exists(), reason="pipeline not run"
)
def test_readme_tables_match_results() -> None:
    """No hand-typed numbers: the generated README blocks equal what results/ produce now."""
    from kapitalkompass.report import facts
    from kapitalkompass.report.markdown import headline

    f = facts.load()
    for name, lang in (("README.md", "en"), ("README.de.md", "de")):
        text = (ROOT / name).read_text()
        assert headline(f, lang) in text, f"{name} is stale: run `task report`"
    assert "TBD" not in (ROOT / "README.md").read_text()


def test_dagster_definitions() -> None:
    from kapitalkompass.mlops.pipeline import defs

    job = defs.resolve_job_def("pipeline")
    keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    assert {"synthetic_bank", "credit", "bayes", "alm", "steering", "drift"} <= keys
    assert job is not None
    checks = defs.resolve_asset_graph().asset_check_keys
    assert len(checks) >= 8


def test_granularity_addon_grows_with_hhi_in_results() -> None:
    p = RESULTS_DIR / "mlops" / "hypotheses_experiments.json"
    if not p.exists():
        pytest.skip("experiments not run")
    rows = json.loads(p.read_text())["granularity"]["portfolios"]
    hhi = np.array([r["hhi_name"] for r in rows])
    addon = np.array([r["addon"] for r in rows])
    order = np.argsort(hhi)
    assert np.all(np.diff(addon[order]) > -0.01)  # monotone up to Monte Carlo noise
