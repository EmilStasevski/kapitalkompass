"""Module 3 cross-checks: BCBS 368 worked examples, zero shock, LCR/NSFR toy cases,
balance-sheet identity of the ladders, survival recovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from kapitalkompass.alm.curves import ZeroCurve, shock_vector
from kapitalkompass.alm.deposits import behavioural_profile, fit_model, simulate_accounts
from kapitalkompass.alm.ftp import ftp_loans, liquidity_premium
from kapitalkompass.alm.irrbb import bullet, delta_nii, eve, floating, run_scenarios
from kapitalkompass.alm.liquidity import hqla_stock, ladder, lcr, nsfr
from kapitalkompass.alm.run import (
    DEPOSITS,
    balance_positions,
    base_curve,
    loan_positions,
    principal_flows,
)
from kapitalkompass.config import SYNTH_DIR, load_config

ACFG = load_config("alm")


def test_bcbs368_short_scalar_example() -> None:
    """Annex 2 example: S_short(3.5) = e^(-3.5/4) = 0.417."""
    d = shock_vector("short_up", [3.5], 0.0, 0.01, 0.0)
    assert d[0] == pytest.approx(0.0041686, abs=1e-6)
    assert round(d[0] * 1e4, 1) == 41.7


def test_bcbs368_rotation_examples() -> None:
    """Annex 2 examples at t = 3.5y with |short| = |long| = 100bp: +25.4bp and -1.6bp."""
    st = shock_vector("steepener", [3.5], 0.0, 0.01, 0.01)[0] * 1e4
    fl = shock_vector("flattener", [3.5], 0.0, 0.01, 0.01)[0] * 1e4
    assert round(st, 1) == 25.4
    assert round(fl, 1) == -1.6


def test_floor_never_raises_base() -> None:
    c = ZeroCurve([1, 10], [-0.005, 0.01])
    s = c.shifted(np.array([-0.02, -0.02]), floor=0.0)
    assert s.r[0] == pytest.approx(-0.005)
    assert s.r[1] == pytest.approx(0.0)


def test_eve_zero_coupon_closed_form() -> None:
    c = ZeroCurve([0.25, 30], [0.03, 0.03])
    p = bullet("z", "asset", 100.0, 5.0, 0.0)
    assert eve([p], c) == pytest.approx(100 * np.exp(-0.15))
    up = c.shifted(np.full(2, 0.01))
    assert eve([p], up) - eve([p], c) == pytest.approx(100 * (np.exp(-0.2) - np.exp(-0.15)))


def test_nii_floating_toy() -> None:
    c = ZeroCurve([0.25, 30], [0.03, 0.03])
    p = floating("f", "asset", 1000.0, 0.25, 0.03)
    up = c.shifted(np.full(2, 0.01))
    assert delta_nii([p], c, up, 1.0) == pytest.approx(1000 * 0.01 * (1 - 0.125))


@pytest.fixture(scope="module")
def bank() -> dict[str, Any]:
    lb = pl.read_parquet(SYNTH_DIR / "loan_book.parquet")
    bsdf = pl.read_parquet(SYNTH_DIR / "balance_sheet.parquet")
    bs = {r["item"]: float(r["amount_eur"]) for r in bsdf.iter_rows(named=True)}
    dcfg = ACFG["deposits"]
    acc = simulate_accounts(dcfg, 7)
    cats = list(dcfg["categories"])
    model = fit_model(acc, cats)
    profiles = {}
    for c in cats:
        lbal = acc.filter(pl.col("category") == c)["log_balance"].to_numpy()
        cap = dcfg["bcbs_caps"][dcfg["categories"][c]["bcbs_class"]]
        profiles[c] = behavioural_profile(model, c, lbal, cap, 1.0, 20.0)
    return {"lb": lb, "bs": bs, "profiles": profiles, "model": model, "acc": acc}


def test_zero_shock_zero_delta(bank: dict[str, Any]) -> None:
    curve, _ = base_curve()
    icfg = ACFG["irrbb"]
    pos = loan_positions(bank["lb"], curve, 0.25) + balance_positions(
        bank["bs"], curve, bank["profiles"], icfg["deposit_beta"]
    )
    sc = run_scenarios(pos, curve, icfg["shocks_bp"], 4.0, 0.0, 1.0)
    assert sc["none"]["delta_eve"] == 0.0
    assert sc["none"]["delta_nii"] == 0.0
    # symmetric linear NII under parallel shocks (floor not binding at current rates)
    assert sc["parallel_up"]["delta_nii"] == pytest.approx(-sc["parallel_down"]["delta_nii"])


def test_hqla_caps_hand_computed() -> None:
    """L1 100, L2A 85 (after 15%), L2B 30 (after 50%): 15% cap -> adj 5; 40% cap binds,
    so HQLA = L1 / 0.6 = 166.67 (LCR30.38-30.39)."""
    s = hqla_stock(100.0, 85.0, 30.0)
    assert s["adj15"] == pytest.approx(5.0)
    assert s["adj40"] == pytest.approx(85 + 30 - 5 - 200 / 3)
    assert s["hqla"] == pytest.approx(100 / 0.6)


def test_lcr_toy() -> None:
    cfg = {
        "hqla": {"cash": {"level": "L1", "haircut": 0.0}},
        "outflow_rates": {"retail_deposits_stable": 0.05},
        "wholesale_lt1y_maturing_30d_share": 1 / 12,
        "inflow_rate_nonfinancial": 0.5,
        "inflow_cap": 0.75,
    }
    r = lcr({"cash": 100.0, "retail_deposits_stable": 1000.0}, 100.0, cfg)
    # outflows 50, inflows min(50, 37.5) = 37.5 -> net 12.5 -> LCR 8
    assert r["outflows"] == pytest.approx(50.0)
    assert r["inflow_cap_binding"]
    assert r["lcr"] == pytest.approx(100 / 12.5)


def test_nsfr_toy() -> None:
    cfg = {
        "asf": {"cet1_equity": 1.0, "retail_deposits_stable": 0.95},
        "rsf": {"loans_ge1y": 0.85, "hqla_l1_gov_bonds": 0.05},
    }
    r = nsfr(
        {"cet1_equity": 100.0, "retail_deposits_stable": 200.0, "hqla_l1_gov_bonds": 100.0},
        0.0,
        300.0,
        cfg,
    )
    assert r["nsfr"] == pytest.approx((100 + 190) / (255 + 5))


def test_ladders_respect_balance_sheet(bank: dict[str, Any]) -> None:
    """Ladder principal = balance-sheet amounts of the rate-/liquidity-relevant lines."""
    bs = bank["bs"]
    flows = principal_flows(bank["lb"], bs)
    contractual = flows + [
        (d, "liability", np.array([1 / 365]), np.array([bs[d]])) for d in DEPOSITS
    ]
    behavioural = list(flows)
    for d in DEPOSITS:
        t, w = bank["profiles"][d].runoff()
        behavioural.append((d, "liability", t, bs[d] * w))
    assets = sum(
        bs[k]
        for k in [
            "loans",
            "hqla_l1_cash_cb",
            "hqla_l1_gov_bonds",
            "hqla_l2a_covered",
            "hqla_l2b_corp",
            "bonds_non_hqla",
        ]
    )
    liabs = sum(
        bs[k]
        for k in [
            *DEPOSITS,
            "wholesale_unsecured_lt1y",
            "covered_bonds_issued",
            "wholesale_unsecured_gt1y",
        ]
    )
    for fl in (contractual, behavioural):
        lad = ladder(fl, ACFG["ladder_buckets_years"])
        assert lad["inflows_eur"].sum() == pytest.approx(assets, rel=1e-9)
        assert lad["outflows_eur"].sum() == pytest.approx(liabs, rel=1e-9)


def test_profiles_meet_bcbs_caps(bank: dict[str, Any]) -> None:
    dcfg = ACFG["deposits"]
    for c, p in bank["profiles"].items():
        cap = dcfg["bcbs_caps"][dcfg["categories"][c]["bcbs_class"]]
        _t, w = p.runoff()
        assert w.sum() == pytest.approx(1.0)
        assert p.core_share <= cap["core_share"] + 1e-12
        assert p.avg_maturity_core <= cap["avg_maturity"] + 1e-9


def test_survival_recovers_truth(bank: dict[str, Any]) -> None:
    coef = bank["model"].aft.params_.loc[("lambda_", "log_balance")]
    assert coef == pytest.approx(ACFG["deposits"]["balance_effect"], abs=0.03)


def test_ftp_margin_identity(bank: dict[str, Any]) -> None:
    curve, _ = base_curve()
    lb = bank["lb"].head(2000)
    f = ftp_loans(lb, curve, 0.009, 3.0, 0.25)
    lp = liquidity_premium(lb["maturity_years"].to_numpy() / 2, 0.009, 3.0)
    expect = (lb["client_margin"].to_numpy() - lp) * lb["ead_eur"].to_numpy()
    assert np.allclose(f["nim_after_ftp_eur"].to_numpy(), expect)
