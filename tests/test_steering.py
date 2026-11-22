"""Modules 4 and 5: allocation additivity, hand-solved LP, constraint feasibility,
traffic lights, hedge."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from kapitalkompass.alm.curves import ZeroCurve
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.results import RESULTS_DIR, read_json, read_table
from kapitalkompass.steering.limits import hedge_notional, lcr_light, light, swap_unit_deltas
from kapitalkompass.steering.optimiser import Problem, check_feasible, solve
from kapitalkompass.steering.raroc import bank_raroc, raroc_table


def tiny(**kw: object) -> Problem:
    base = dict(
        names=["A", "B"],
        profit=np.array([10.0, 6.0]),
        capital=np.array([50.0, 20.0]),
        ead=np.array([1000.0, 1000.0]),
        sector=np.array([0, 0]),
        sector_names=["s"],
        sector_max=np.array([1.0]),
        hurdle=0.1,
        capital_budget=100.0,
        lo=0.5,
        hi=2.0,
    )
    base.update(kw)
    return Problem(**base)  # type: ignore[arg-type]


def test_hand_solved_lp() -> None:
    """EVA/capital: A 5/50 = 0.10, B 4/20 = 0.20 -> fill B to its bound (x_B = 2, capital
    40), remaining 60 to A (x_A = 1.2). Objective 5*1.2 + 4*2 = 14; shadow price of
    capital = marginal EVA per EUR of A's capital = 0.10."""
    sol = solve(tiny())
    assert sol.status == "optimal"
    assert sol.x == pytest.approx([1.2, 2.0], abs=1e-6)
    assert sol.objective == pytest.approx(14.0, abs=1e-6)
    assert sol.constraints["capital_budget"]["shadow_price"] == pytest.approx(0.1, abs=1e-6)
    assert sol.constraints["capital_budget"]["binding"]


def test_lp_sector_limit_binds() -> None:
    """Two sectors, B's sector capped at 50% of volume: x_B <= x_A."""
    sol = solve(
        tiny(sector=np.array([0, 1]), sector_names=["a", "b"], sector_max=np.array([1.0, 0.5]))
    )
    assert sol.x[1] <= sol.x[0] + 1e-6
    # with x_A = x_B = x: 70 x <= 100 -> x = 10/7
    assert sol.x == pytest.approx([10 / 7, 10 / 7], abs=1e-5)


def test_optimiser_never_violates_constraints() -> None:
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(40):
        n = 12
        p = Problem(
            names=[str(i) for i in range(n)],
            profit=rng.uniform(-1, 10, n) * 1e6,
            capital=rng.uniform(1, 40, n) * 1e6,
            ead=rng.uniform(100, 900, n) * 1e6,
            sector=np.repeat([0, 1, 2], 4),
            sector_names=["a", "b", "c"],
            sector_max=rng.uniform(0.34, 0.6, 3),
            hurdle=0.1,
            capital_budget=0.0,
            lo=0.8,
            hi=1.2,
            hqla0=2e9,
            net_out=1e9,
            lcr_floor=1.3,
            rsf0=5e9,
            asf=6.5e9,
            rsf_w=rng.uniform(0.5, 0.85, n),
            nsfr_floor=1.1,
        )
        p.capital_budget = float(p.capital.sum())
        sol = solve(p)
        if sol.status != "optimal":
            continue
        solved += 1
        assert check_feasible(p, sol.x, tol=1e-6) == []
    assert solved >= 30  # the check is not vacuous


def test_traffic_lights() -> None:
    assert light(0.5, 0.8, 1.0) == "green"
    assert light(0.8, 0.8, 1.0) == "amber"
    assert light(1.0, 0.8, 1.0) == "amber"  # at the limit, not beyond
    assert light(1.01, 0.8, 1.0) == "red"
    assert lcr_light(1.5, 1.3, 1.1) == "green"
    assert lcr_light(1.2, 1.3, 1.1) == "amber"
    assert lcr_light(1.05, 1.3, 1.1) == "red"


def test_hedge_restores_eve_limit() -> None:
    c = ZeroCurve([0.25, 1, 5, 10, 30], [0.027, 0.03, 0.032, 0.034, 0.037])
    unit = {
        k: v["eve"]
        for k, v in swap_unit_deltas(
            c, 5.0, {"parallel": 200, "short": 250, "long": 100}, 4.0, 0.0
        ).items()
    }
    assert unit["parallel_down"] > 0 > unit["parallel_up"]  # receiver gains when rates fall
    bank = {
        "parallel_up": 50.0,
        "parallel_down": -150.0,
        "steepener": 20.0,
        "flattener": -30.0,
        "short_up": 5.0,
        "short_down": -5.0,
    }
    res = hedge_notional(bank, unit, 100.0)
    assert res["feasible"] and res["notional_eur"] > 0
    assert res["worst_after_eur"] >= -100.0 - 1e-6


@pytest.mark.skipif(
    not (RESULTS_DIR / "steering" / "raroc.csv").exists(), reason="pipeline not run"
)
def test_allocation_additivity_and_raroc_identity() -> None:
    """Euler EC per segment sums to total EC; RAROC = numerator / EC row by row."""
    credit = read_json("credit/summary.json")
    cap = read_table("credit/segment_capital.csv")
    assert cap["economic_capital_eur"].sum() == pytest.approx(
        credit["economic_capital_eur"], rel=1e-8
    )
    lb = pl.read_parquet(SYNTH_DIR / "loan_book.parquet")
    t = raroc_table(
        lb, read_table("alm/ftp_segments.csv"), cap, load_config("steering")["hurdle_rate"]
    )
    num = t["nim_after_ftp_eur"] + t["fees_eur"] - t["opex_eur"] - t["el_eur"]
    assert np.allclose(t["raroc"].to_numpy(), (num / t["economic_capital_eur"]).to_numpy())
    assert bank_raroc(t) == pytest.approx(
        float(num.to_numpy().sum() / t["economic_capital_eur"].to_numpy().sum())
    )
    opt = read_json("steering/optimisation.json")
    assert opt["feasibility_violations"] == []
    assert opt["capital_after_resimulated_eur"] <= opt["capital_before_eur"] * (1 + 1e-3)
