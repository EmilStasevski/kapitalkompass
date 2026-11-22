"""Day 1: determinism, data contracts and generator cross-checks."""

from __future__ import annotations

import numpy as np
import pandera.errors
import polars as pl
import pytest

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.schemas import BalanceSheet, LoanBook
from kapitalkompass.data.synth import build, frame_hash


@pytest.fixture(scope="module")
def bank() -> dict[str, pl.DataFrame]:
    return build(write=False)


def test_deterministic(bank: dict[str, pl.DataFrame]) -> None:
    again = build(write=False)
    for name in bank:
        assert frame_hash(bank[name]) == frame_hash(again[name])


def test_matches_locked_hashes(bank: dict[str, pl.DataFrame]) -> None:
    """Locks bit-for-bit reproducibility across machines (HASHES is committed)."""
    locked: dict[str, str] = {}
    for line in (SYNTH_DIR / "HASHES").read_text().splitlines():
        digest, name = line.split("  ")
        locked[name] = digest
    for name, df in bank.items():
        assert frame_hash(df) == locked[name], f"{name} changed; regenerate HASHES deliberately"


def test_seed_changes_output(bank: dict[str, pl.DataFrame]) -> None:
    other = build(seed=1, write=False)
    assert frame_hash(other["loan_book"]) != frame_hash(bank["loan_book"])


def test_schemas_pass(bank: dict[str, pl.DataFrame]) -> None:
    LoanBook.validate(bank["loan_book"])
    BalanceSheet.validate(bank["balance_sheet"])


def test_schema_rejects_negative_ead(bank: dict[str, pl.DataFrame]) -> None:
    bad = bank["loan_book"].with_columns(
        ead_eur=pl.when(pl.col("obligor_id") == 0).then(-1.0).otherwise(pl.col("ead_eur"))
    )
    with pytest.raises(pandera.errors.SchemaError):
        LoanBook.validate(bad)


def test_schema_rejects_wrong_segment(bank: dict[str, pl.DataFrame]) -> None:
    bad = bank["loan_book"].with_columns(
        segment=pl.when(pl.col("obligor_id") == 0)
        .then(pl.lit("trade_1"))
        .otherwise(pl.col("segment"))
    )
    if bank["loan_book"]["segment"][0] == "trade_1":
        bad = bad.with_columns(segment=pl.lit("trade_2"))
    with pytest.raises(pandera.errors.SchemaError):
        LoanBook.validate(bad)


def test_balance_identity_exact(bank: dict[str, pl.DataFrame]) -> None:
    bs = bank["balance_sheet"]
    assets = bs.filter(pl.col("side") == "asset")["amount_eur"].sum()
    other = bs.filter(pl.col("side") != "asset")["amount_eur"].sum()
    assert assets == other
    broken = bs.with_columns(
        amount_eur=pl.when(pl.col("item") == "loans")
        .then(pl.col("amount_eur") + 1)
        .otherwise(pl.col("amount_eur"))
    )
    with pytest.raises(pandera.errors.SchemaError):
        BalanceSheet.validate(broken)


def test_loans_tie_to_balance_sheet(bank: dict[str, pl.DataFrame]) -> None:
    loans = bank["balance_sheet"].filter(pl.col("item") == "loans")["amount_eur"].item()
    assert loans == round(bank["loan_book"]["ead_eur"].sum())


def test_sample_moments_match_config(bank: dict[str, pl.DataFrame]) -> None:
    """Analytic cross-check: shares and mean LGD within 4 standard errors of config."""
    cfg = load_config("bank")["loan_book"]
    lb = bank["loan_book"]
    n = lb.height
    for s, spec in cfg["sectors"].items():
        share = (lb["sector"] == s).mean()
        se = np.sqrt(spec["share"] * (1 - spec["share"]) / n)
        assert abs(share - spec["share"]) < 4 * se, s
        lgd = lb.filter(pl.col("sector") == s)["lgd"]
        m, c = spec["lgd_mean"], cfg["lgd_concentration"]
        se_lgd = np.sqrt(m * (1 - m) / (c + 1) / lgd.len())  # beta variance
        assert abs(lgd.mean() - m) < 4 * se_lgd + 0.002, s  # +clip bias allowance
    for g, spec in cfg["grades"].items():
        share = (lb["grade"] == int(g)).mean()
        se = np.sqrt(spec["share"] * (1 - spec["share"]) / n)
        assert abs(share - spec["share"]) < 4 * se, g


def test_large_names(bank: dict[str, pl.DataFrame]) -> None:
    cfg = load_config("bank")["loan_book"]["exposure"]
    large = bank["loan_book"].filter(pl.col("is_large"))
    assert large.height == cfg["n_large"]
    assert large["ead_eur"].min() >= cfg["large_pareto_min_eur"]
    assert large["ead_eur"].max() <= cfg["large_cap_eur"]
    assert set(large["sector"].unique()) <= set(cfg["large_sectors"])
