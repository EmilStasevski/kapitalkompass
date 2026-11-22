"""Modules 4 and 5 pipeline step: RAROC, constrained re-allocation, limits, hedge action."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from kapitalkompass.alm.run import base_curve
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.credit.portfolio import risk_measures, simulate
from kapitalkompass.credit.run import build_portfolio, resolve_device
from kapitalkompass.data.schemas import SECTORS, SEGMENTS, BalanceSheet, LoanBook
from kapitalkompass.results import read_json, read_table, write_json, write_table
from kapitalkompass.steering.limits import hedge_notional, limit_table, swap_unit_deltas
from kapitalkompass.steering.optimiser import Problem, check_feasible, solve
from kapitalkompass.steering.raroc import bank_raroc, raroc_table


def resimulated_ec(
    lb: pl.DataFrame, x_by_seg: dict[str, float], scenarios: int, device: str
) -> float:
    mcfg = load_config("model")
    scaled = lb.with_columns(
        ead_eur=pl.col("ead_eur")
        * pl.col("segment").replace_strict(x_by_seg, return_dtype=pl.Float64)
    )
    pf = build_portfolio(scaled, mcfg["credit"])
    res = simulate(
        pf,
        scenarios,
        int(mcfg["seed"]) + 10,
        device,
        mcfg["credit"]["chunk_scenarios"],
        mcfg["credit"]["obligor_block"],
    )
    rm = risk_measures(pf, res, float(mcfg["credit"]["confidence"]))
    return rm.es - rm.el_analytic


def build_problem(
    t: pl.DataFrame, lb: pl.DataFrame, alm: dict[str, Any], scfg: dict[str, Any], budget: float
) -> Problem:
    ocfg = scfg["optimiser"]
    acfg = load_config("alm")
    rsf = (
        lb.with_columns(
            rsf=pl.when(pl.col("maturity_years") < 1)
            .then(acfg["nsfr"]["rsf"]["loans_lt1y"])
            .otherwise(acfg["nsfr"]["rsf"]["loans_ge1y"])
        )
        .group_by("segment")
        .agg(rsf_w=(pl.col("rsf") * pl.col("ead_eur")).sum() / pl.col("ead_eur").sum())
    )
    t = t.join(rsf, on="segment").sort("segment")
    # Funding source (L1 cash) carries 0% RSF, so the swap changes RSF by rsf_w per EUR.
    rsf_w = t["rsf_w"].to_numpy() - acfg["nsfr"]["rsf"][ocfg["funding"]]
    return Problem(
        names=t["segment"].to_list(),
        profit=t["risk_adjusted_profit_eur"].to_numpy(),
        capital=t["economic_capital_eur"].to_numpy(),
        ead=t["ead_eur"].to_numpy(),
        sector=np.array([SECTORS.index(s) for s in t["sector"].to_list()]),
        sector_names=list(SECTORS),
        sector_max=np.array([ocfg["sector_share_max"][s] for s in SECTORS]),
        hurdle=float(scfg["hurdle_rate"]),
        capital_budget=budget,
        lo=float(ocfg["growth_bounds"][0]),
        hi=float(ocfg["growth_bounds"][1]),
        hqla0=float(alm["lcr"]["hqla"]),
        net_out=float(alm["lcr"]["net_outflows"]),
        lcr_floor=float(ocfg["lcr_floor"]),
        rsf0=float(alm["nsfr"]["rsf"]),
        asf=float(alm["nsfr"]["asf"]),
        rsf_w=rsf_w,
        nsfr_floor=float(ocfg["nsfr_floor"]),
    )


def run(device: str | None = None) -> dict[str, Any]:
    scfg = load_config("steering")
    acfg = load_config("alm")
    dev = resolve_device(device)
    lb = LoanBook.validate(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"))
    bsdf = BalanceSheet.validate(pl.read_parquet(SYNTH_DIR / "balance_sheet.parquet"))
    bs = {r["item"]: float(r["amount_eur"]) for r in bsdf.iter_rows(named=True)}
    capital = read_table("credit/segment_capital.csv")
    credit = read_json("credit/summary.json")
    alm = read_json("alm/summary.json")
    ftp = read_table("alm/ftp_segments.csv")
    hurdle = float(scfg["hurdle_rate"])

    # ---- module 4: RAROC
    t = raroc_table(lb, ftp, capital, hurdle)
    write_table("steering/raroc.csv", t)
    r_bank = bank_raroc(t)
    total_ec = float(t["economic_capital_eur"].to_numpy().sum())
    euler_gap = abs(total_ec - credit["economic_capital_eur"]) / credit["economic_capital_eur"]

    # ---- module 5: optimiser with ex-post re-simulation of the capital budget
    budget = total_ec
    iterations = []
    n_resim = int(scfg["optimiser"]["resimulate_scenarios"])
    for it in range(3):
        prob = build_problem(t, lb, alm, scfg, budget)
        sol = solve(prob)
        x_seg = dict(zip(prob.names, sol.x.tolist(), strict=True))
        ec_new = resimulated_ec(lb, x_seg, n_resim, dev)
        iterations.append(
            {
                "iteration": it,
                "budget_eur": budget,
                "linear_ec_eur": float(prob.capital @ sol.x),
                "resimulated_ec_eur": ec_new,
            }
        )
        if ec_new <= total_ec * (1 + 1e-3):
            break
        budget *= total_ec / ec_new  # tighten the linear budget by the non-linearity and resolve
    feas = check_feasible(prob, sol.x)
    profit_after = float(prob.profit @ sol.x)
    alloc = (
        t.select("segment", "sector", "ead_eur", "raroc", "economic_capital_eur")
        .with_columns(
            multiplier=sol.x,
            ead_after_eur=pl.col("ead_eur") * sol.x,
        )
        .with_columns(change_eur=pl.col("ead_after_eur") - pl.col("ead_eur"))
    )
    write_table("steering/allocation.csv", alloc)
    d_ead = float(prob.ead @ (sol.x - 1))
    lcr_after = (prob.hqla0 - d_ead) / prob.net_out if prob.hqla0 and prob.net_out else None
    nsfr_after = (
        prob.asf / (prob.rsf0 + float((prob.rsf_w * prob.ead) @ (sol.x - 1)))
        if prob.asf and prob.rsf0 and prob.rsf_w is not None
        else None
    )
    sector_after = {
        s: float((prob.ead * (prob.sector == k)) @ sol.x / (prob.ead @ sol.x))
        for k, s in enumerate(SECTORS)
    }
    binding = [k for k, v in sol.constraints.items() if v["binding"]]
    opt = {
        "status": sol.status,
        "objective": "risk-adjusted profit minus hurdle x EC (EVA)",
        "raroc_before": r_bank,
        "raroc_after_linear": profit_after / float(prob.capital @ sol.x),
        "raroc_after_resimulated": profit_after / iterations[-1]["resimulated_ec_eur"],
        "eva_before_eur": float(t["eva_eur"].to_numpy().sum()),
        "eva_after_eur": profit_after - hurdle * iterations[-1]["resimulated_ec_eur"],
        "profit_before_eur": float(prob.profit.sum()),
        "profit_after_eur": profit_after,
        "capital_before_eur": total_ec,
        "capital_after_resimulated_eur": iterations[-1]["resimulated_ec_eur"],
        "capital_iterations": iterations,
        "total_ead_before_eur": float(prob.ead.sum()),
        "total_ead_after_eur": float(prob.ead @ sol.x),
        "lcr_after": lcr_after,
        "nsfr_after": nsfr_after,
        "sector_share_after": sector_after,
        "binding_constraints": binding,
        "constraints": sol.constraints,
        "feasibility_violations": feas,
    }
    write_json("steering/optimisation.json", opt)

    # ---- limit system (before and after re-allocation)
    ir = alm["irrbb"]
    sector_before = {
        s: float(lb.filter(pl.col("sector") == s)["ead_eur"].sum()) / float(lb["ead_eur"].sum())
        for s in SECTORS
    }
    common = {
        "risk_cover": bs[scfg["limits"]["risk_cover_item"]],
        "tier1": alm["tier1_eur"],
        "eve_loss": max(0.0, -ir["worst_delta_eve_eur"]),
        "nii_loss": max(0.0, -ir["worst_delta_nii_eur"]),
        "max_name": float(lb["ead_eur"].to_numpy().max()),
    }
    before = limit_table(
        {
            **common,
            "credit_ec": total_ec,
            "max_sector_share": max(sector_before.values()),
            "lcr": alm["lcr"]["lcr"],
        },
        scfg["limits"],
    )
    x_max_name = float(
        lb.with_columns(
            x=pl.col("segment").replace_strict(
                dict(zip(prob.names, sol.x.tolist(), strict=True)), return_dtype=pl.Float64
            )
        )
        .select((pl.col("ead_eur") * pl.col("x")).max())
        .item()
    )
    after = limit_table(
        {
            **common,
            "max_name": x_max_name,
            "credit_ec": iterations[-1]["resimulated_ec_eur"],
            "max_sector_share": max(sector_after.values()),
            "lcr": lcr_after or 0.0,
        },
        scfg["limits"],
    )
    write_table("steering/limits_before.csv", before)
    write_table("steering/limits_after.csv", after)

    # ---- IRRBB hedge recommendation for the EVE limit
    curve, _ = base_curve()
    icfg = acfg["irrbb"]
    unit_all = swap_unit_deltas(
        curve, scfg["hedge"]["tenor_years"], icfg["shocks_bp"], icfg["decay_x"], icfg["floor"]
    )
    unit = {k: v["eve"] for k, v in unit_all.items()}
    bank_deve = {k: v["delta_eve"] for k, v in ir["scenarios"].items() if k != "none"}
    max_loss = (
        scfg["hedge"]["target_share_of_limit"]
        * scfg["limits"]["items"]["irrbb_eve"]["limit_share_of_cover"]
        * alm["tier1_eur"]
    )
    hedge = hedge_notional(bank_deve, unit, max_loss)
    h = float(hedge.get("notional_eur", 0.0))
    nii_after = {
        k: v["delta_nii"] + h * unit_all[k]["nii"]
        for k, v in ir["scenarios"].items()
        if k != "none"
    }
    hedge.update(
        {
            "tenor_years": scfg["hedge"]["tenor_years"],
            "target_max_loss_eur": max_loss,
            "unit_delta_eve": unit,
            "unit_delta_nii": {k: v["nii"] for k, v in unit_all.items()},
            "delta_nii_after": nii_after,
            "worst_nii_after_eur": min(nii_after.values()),
        }
    )
    write_json("steering/hedge.json", hedge)

    # ---- limits after management actions: re-allocation + hedge + single-name sell-down
    name_limit = scfg["limits"]["items"]["single_name"]["limit"] * alm["tier1_eur"]
    sell_down = max(0.0, x_max_name - name_limit)
    after_actions = limit_table(
        {
            **common,
            "max_name": min(x_max_name, name_limit),
            "credit_ec": iterations[-1]["resimulated_ec_eur"],
            "max_sector_share": max(sector_after.values()),
            "lcr": lcr_after or 0.0,
            "eve_loss": max(0.0, -float(hedge.get("worst_after_eur", ir["worst_delta_eve_eur"]))),
            "nii_loss": max(0.0, -min(nii_after.values())),
        },
        scfg["limits"],
    )
    write_table("steering/limits_after_actions.csv", after_actions)
    actions = {
        "reallocation": {
            "raroc_before": r_bank,
            "raroc_after": opt["raroc_after_resimulated"],
            "largest_increases": alloc.sort("change_eur", descending=True)
            .head(3)["segment"]
            .to_list(),
            "largest_decreases": alloc.sort("change_eur").head(3)["segment"].to_list(),
        },
        "irrbb_hedge": {
            "instrument": "receive-fixed swap",
            "tenor_years": scfg["hedge"]["tenor_years"],
            "notional_eur": h,
        },
        "single_name_sell_down_eur": sell_down,
        "note": "Sell-down capital relief is not re-simulated (small relative to the book).",
    }
    write_json("steering/actions.json", actions)

    def counts(tab: pl.DataFrame) -> dict[str, int]:
        return {c: int((tab["status"] == c).sum()) for c in ("green", "amber", "red")}

    summary = {
        "hurdle_rate": hurdle,
        "bank_raroc": r_bank,
        "segments": len(SEGMENTS),
        "segments_below_hurdle": int(t["below_hurdle"].sum()),
        "segments_below_hurdle_names": t.filter("below_hurdle")["segment"].to_list(),
        "euler_allocation_sums_to_total": euler_gap < 1e-9,
        "euler_relative_gap": euler_gap,
        "limits_before": counts(before),
        "limits_after": counts(after),
        "red_before": before.filter(pl.col("status") == "red")["limit"].to_list(),
        "red_after": after.filter(pl.col("status") == "red")["limit"].to_list(),
        "limits_after_actions": counts(after_actions),
        "red_after_actions": after_actions.filter(pl.col("status") == "red")["limit"].to_list(),
        "single_name_sell_down_eur": sell_down,
        "optimiser": {
            k: opt[k]
            for k in (
                "raroc_before",
                "raroc_after_resimulated",
                "binding_constraints",
                "feasibility_violations",
            )
        },
        "hedge": {
            k: hedge.get(k)
            for k in (
                "feasible",
                "notional_eur",
                "direction",
                "worst_after_scenario",
                "worst_after_eur",
            )
        },
    }
    write_json("steering/summary.json", summary)
    print(
        f"steering: RAROC {r_bank:.1%} (hurdle {hurdle:.0%}), "
        f"{summary['segments_below_hurdle']} below hurdle; optimiser "
        f"{opt['raroc_before']:.1%} -> {opt['raroc_after_resimulated']:.1%}, binding {binding}"
    )
    print(
        f"limits {summary['limits_before']} -> {summary['limits_after']} -> after actions "
        f"{summary['limits_after_actions']}; hedge {h / 1e6:.0f}m, sell-down {sell_down / 1e6:.1f}m"
    )
    return summary


if __name__ == "__main__":
    run()
