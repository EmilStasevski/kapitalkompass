"""Allowlisted agent tools with Pydantic argument schemas.

Every tool reads the pipeline results (or re-solves the LP) and returns JSON-serialisable
values in reporting units (Mio. EUR, percent), already rounded, so the LLM copies numbers rather
than computing them. ``dispatch`` is the only entry point: names outside ``TOOLS`` and arguments
that fail validation are rejected before any code runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kapitalkompass.alm.curves import SCENARIOS
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.schemas import SECTORS, SEGMENTS
from kapitalkompass.results import read_json, read_table

if TYPE_CHECKING:
    Segment = str
    Scenario = str
else:  # enums built from the data contract, so the JSON schema constrains decoding
    Segment = Literal[tuple([*SEGMENTS, "bank"])]
    Scenario = Literal[tuple(SCENARIOS)]


def mio(x: float) -> float:
    return round(float(x) / 1e6, 2)


def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetRarocArgs(Strict):
    segment: Segment = Field(description="Segment (sector_grade) or 'bank' for the whole bank")


class GetLimitsArgs(Strict):
    state: Literal["current", "after_reallocation", "after_actions"] = Field(
        description="current book, after optimiser re-allocation, or after hedge + sell-down"
    )


class RunShockArgs(Strict):
    name: Scenario = Field(description="BCBS 368 / EU RTS 2024/856 supervisory shock scenario")


class OptimiserConstraints(Strict):
    hurdle_rate_pct: float | None = Field(None, ge=0.0, le=30.0)
    growth_lower: float | None = Field(None, ge=0.5, le=1.0)
    growth_upper: float | None = Field(None, ge=1.0, le=1.5)
    capital_budget_change_pct: float | None = Field(None, ge=-30.0, le=30.0)
    lcr_floor_pct: float | None = Field(None, ge=100.0, le=300.0)
    nsfr_floor_pct: float | None = Field(None, ge=100.0, le=200.0)
    sector_max_real_estate_pct: float | None = Field(None, ge=0.0, le=100.0)
    sector_max_trade_pct: float | None = Field(None, ge=0.0, le=100.0)
    sector_max_manufacturing_pct: float | None = Field(None, ge=0.0, le=100.0)


class RunOptimiserArgs(Strict):
    constraints: OptimiserConstraints


class SearchRulesArgs(Strict):
    query: str = Field(min_length=3, max_length=200)


class ToolError(Exception):
    pass


# ----------------------------------------------------------------------------- tools


def get_raroc(a: GetRarocArgs) -> dict[str, Any]:
    t = read_table("steering/raroc.csv").sort("raroc", descending=True)
    s = read_json("steering/summary.json")
    credit = read_json("credit/summary.json")
    hurdle = float(s["hurdle_rate"])
    if a.segment == "bank":
        return {
            "segment": "bank",
            "bank_raroc_pct": pct(s["bank_raroc"]),
            "hurdle_rate_pct": pct(hurdle),
            "economic_capital_mio_eur": mio(t["economic_capital_eur"].to_numpy().sum()),
            "risk_adjusted_profit_mio_eur": mio(t["risk_adjusted_profit_eur"].to_numpy().sum()),
            "eva_mio_eur": mio(t["eva_eur"].to_numpy().sum()),
            "expected_loss_mio_eur": mio(credit["el_analytic_eur"]),
            "es_confidence_pct": pct(credit["confidence"]),
            "segments_total": int(s["segments"]),
            "segments_below_hurdle": int(s["segments_below_hurdle"]),
            "segments_below_hurdle_names": list(s["segments_below_hurdle_names"]),
            "segment_raroc_pct": {r["segment"]: pct(r["raroc"]) for r in t.iter_rows(named=True)},
        }
    rows = t.to_dicts()
    rank = next(i for i, r in enumerate(rows, 1) if r["segment"] == a.segment)
    r = rows[rank - 1]
    return {
        "segment": a.segment,
        "sector": r["sector"],
        "raroc_pct": pct(r["raroc"]),
        "hurdle_rate_pct": pct(hurdle),
        "below_hurdle": bool(r["below_hurdle"]),
        "rank_by_raroc": rank,
        "segments_total": len(rows),
        "ead_mio_eur": mio(r["ead_eur"]),
        "economic_capital_mio_eur": mio(r["economic_capital_eur"]),
        "risk_adjusted_profit_mio_eur": mio(r["risk_adjusted_profit_eur"]),
        "nim_after_ftp_mio_eur": mio(r["nim_after_ftp_eur"]),
        "expected_loss_mio_eur": mio(r["el_eur"]),
        "eva_mio_eur": mio(r["eva_eur"]),
    }


LIMIT_FILES = {
    "current": "steering/limits_before.csv",
    "after_reallocation": "steering/limits_after.csv",
    "after_actions": "steering/limits_after_actions.csv",
}


def get_limits(a: GetLimitsArgs) -> dict[str, Any]:
    t = read_table(LIMIT_FILES[a.state])
    items = []
    lcr_cfg = load_config("steering")["limits"]["lcr"]
    for r in t.iter_rows(named=True):
        money = r["unit"] == "EUR"
        if r["limit"] == "lcr":  # higher is better: report the ratio, not a utilisation
            items.append(
                {
                    "limit": "lcr",
                    "description": "Liquidity coverage ratio (LCR) of the bank vs internal floor",
                    "status": r["status"],
                    "lcr_pct": pct(r["exposure"]),
                    "internal_floor_green_from_pct": pct(lcr_cfg["green_from"]),
                    "red_below_pct": pct(lcr_cfg["amber_from"]),
                    "regulatory_minimum_pct": 100.0,
                    "escalation": r["escalation"],
                }
            )
            continue
        items.append(
            {
                "limit": r["limit"],
                "description": r["description"],
                "status": r["status"],
                "utilisation_pct": pct(r["utilisation"]),
                ("exposure_mio_eur" if money else "exposure_pct"): (
                    mio(r["exposure"]) if money else pct(r["exposure"])
                ),
                ("limit_mio_eur" if money else "limit_pct"): (
                    mio(r["limit_value"]) if money else pct(r["limit_value"])
                ),
                "escalation": r["escalation"],
            }
        )
    out: dict[str, Any] = {
        "state": a.state,
        "counts": {k: sum(i["status"] == k for i in items) for k in ("green", "amber", "red")},
        "limits": items,
    }
    if a.state == "after_actions":
        act = read_json("steering/actions.json")
        hedge = read_json("steering/hedge.json")
        out["actions"] = {
            "irrbb_hedge": "receive-fixed interest rate swap",
            "hedge_notional_mio_eur": mio(act["irrbb_hedge"]["notional_eur"]),
            "hedge_tenor_years": float(act["irrbb_hedge"]["tenor_years"]),
            "worst_delta_eve_after_hedge_mio_eur": mio(min(hedge["delta_eve_after"].values())),
            "single_name_sell_down_mio_eur": mio(act["single_name_sell_down_eur"]),
        }
    return out


def run_shock(a: RunShockArgs) -> dict[str, Any]:
    alm = read_json("alm/summary.json")
    ir = alm["irrbb"]
    tier1 = float(alm["tier1_eur"])
    sc = ir["scenarios"][a.name]
    worst = ir["worst_eve_scenario"]
    eve_pct, nii_pct = -sc["delta_eve"] / tier1, -sc["delta_nii"] / tier1
    eve_thr, nii_thr = ir["eve_threshold_pct_tier1"], ir["nii_threshold_pct_tier1"]
    eve_txt = (
        f"Szenario {a.name}: EVE-Verlust {pct(max(eve_pct, 0))} % des Kernkapitals "
        + ("ÜBERSCHREITET" if eve_pct > eve_thr else "unterschreitet")
        + f" die Ausreißerschwelle von {pct(eve_thr)} %."
    )
    nii_txt = (
        f"Szenario {a.name}: NII-Rückgang {pct(max(nii_pct, 0))} % des Kernkapitals "
        + ("ÜBERSCHREITET" if nii_pct > nii_thr else "unterschreitet")
        + f" die Schwelle 'large decline' von {pct(nii_thr)} %."
    )
    return {
        "scenario": a.name,
        "eve_outlier_test": eve_txt,
        "nii_large_decline_test": nii_txt,
        "bank_is_eve_outlier": bool(ir["eve_outlier"]),
        "bank_has_nii_large_decline": bool(ir["nii_flag"]),
        "delta_eve_mio_eur": mio(sc["delta_eve"]),
        "delta_eve_pct_tier1": pct(sc["delta_eve"] / tier1),
        "delta_nii_mio_eur": mio(sc["delta_nii"]),
        "delta_nii_pct_tier1": pct(sc["delta_nii"] / tier1),
        "tier1_mio_eur": mio(tier1),
        "eve_outlier_threshold_pct_tier1": pct(ir["eve_threshold_pct_tier1"]),
        "nii_large_decline_threshold_pct_tier1": pct(ir["nii_threshold_pct_tier1"]),
        "eve_outlier_in_this_scenario": bool(
            -sc["delta_eve"] / tier1 > ir["eve_threshold_pct_tier1"]
        ),
        "nii_large_decline_in_this_scenario": bool(
            -sc["delta_nii"] / tier1 > ir["nii_threshold_pct_tier1"]
        ),
        "worst_eve_scenario": worst,
        "worst_delta_eve_mio_eur": mio(ir["worst_delta_eve_eur"]),
        "curve_date": alm["curve_date"],
    }


def run_optimiser(a: RunOptimiserArgs) -> dict[str, Any]:
    """Re-solve the steering LP with modified constraints (linear Euler EC, no re-simulation)."""
    from kapitalkompass.steering.optimiser import check_feasible, solve
    from kapitalkompass.steering.run import build_problem

    c = a.constraints
    scfg = load_config("steering")
    ocfg = dict(scfg["optimiser"])
    if c.growth_lower is not None or c.growth_upper is not None:
        lo, hi = ocfg["growth_bounds"]
        ocfg["growth_bounds"] = [c.growth_lower or lo, c.growth_upper or hi]
    if c.lcr_floor_pct is not None:
        ocfg["lcr_floor"] = c.lcr_floor_pct / 100
    if c.nsfr_floor_pct is not None:
        ocfg["nsfr_floor"] = c.nsfr_floor_pct / 100
    smax = dict(ocfg["sector_share_max"])
    for s in SECTORS:
        v = getattr(c, f"sector_max_{s}_pct")
        if v is not None:
            smax[s] = v / 100
    ocfg["sector_share_max"] = smax
    cfg = {**scfg, "optimiser": ocfg}
    if c.hurdle_rate_pct is not None:
        cfg["hurdle_rate"] = c.hurdle_rate_pct / 100
    t = read_table("steering/raroc.csv")
    lb = pl.read_parquet(SYNTH_DIR / "loan_book.parquet")
    alm = read_json("alm/summary.json")
    ec0 = float(t["economic_capital_eur"].to_numpy().sum())
    budget = ec0 * (1 + (c.capital_budget_change_pct or 0.0) / 100)
    prob = build_problem(t, lb, alm, cfg, budget)
    sol = solve(prob)
    base: dict[str, Any] = {
        "constraints_applied": c.model_dump(exclude_none=True),
        "status": sol.status,
        "capital_model": "linear Euler EC (not re-simulated)",
    }
    if not np.all(np.isfinite(sol.x)):
        return {**base, "feasible": False}
    x = sol.x
    profit = float(prob.profit @ x)
    cap = float(prob.capital @ x)
    d_ead = float(prob.ead @ (x - 1))
    change = sorted(zip(prob.names, (prob.ead * (x - 1)).tolist(), strict=True), key=lambda z: z[1])
    return {
        **base,
        "feasible": not check_feasible(prob, x),
        "hurdle_rate_pct": pct(prob.hurdle),
        "raroc_before_pct": pct(prob.profit.sum() / prob.capital.sum()),
        "raroc_after_pct": pct(profit / cap),
        "raroc_change_pp": round(
            pct(profit / cap) - pct(prob.profit.sum() / prob.capital.sum()), 2
        ),
        "eva_before_mio_eur": mio((prob.profit - prob.hurdle * prob.capital).sum()),
        "eva_after_mio_eur": mio(profit - prob.hurdle * cap),
        "economic_capital_after_mio_eur": mio(cap),
        "capital_budget_mio_eur": mio(budget),
        "total_ead_change_mio_eur": mio(d_ead),
        "lcr_after_pct": pct((prob.hqla0 - d_ead) / prob.net_out)
        if prob.hqla0 and prob.net_out
        else None,
        "binding_constraints": [k for k, v in sol.constraints.items() if v["binding"]],
        "capital_shadow_price_eva_per_eur": round(
            float(sol.constraints["capital_budget"]["shadow_price"]), 4
        ),
        "largest_decreases_mio_eur": {n: mio(v) for n, v in change[:3]},
        "largest_increases_mio_eur": {n: mio(v) for n, v in change[::-1][:3]},
    }


def search_rules(a: SearchRulesArgs) -> dict[str, Any]:
    from kapitalkompass.agent.index import search

    snip = int(load_config("agent")["index"]["snippet_chars"])
    return {
        "query": a.query,
        "chunks": [
            {"id": h["id"], "ref": h["ref"], "text": h["text"][:snip]} for h in search(a.query)
        ],
    }


@dataclass(frozen=True)
class Tool:
    name: str
    args: type[Strict]
    fn: Callable[[Any], dict[str, Any]]
    description: str


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("get_raroc", GetRarocArgs, get_raroc, "RAROC, EC, EVA of one segment or the bank"),
        Tool(
            "get_limits", GetLimitsArgs, get_limits, "Limit utilisation, traffic lights, escalation"
        ),
        Tool("run_shock", RunShockArgs, run_shock, "Delta EVE / NII under one supervisory shock"),
        Tool("run_optimiser", RunOptimiserArgs, run_optimiser, "Re-solve the re-allocation LP"),
        Tool("search_rules", SearchRulesArgs, search_rules, "Search CRR/CRD/EBA/BCBS/MaRisk"),
    ]
}


@dataclass
class Call:
    tool: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0


def dispatch(name: str, args: dict[str, Any]) -> Call:
    """Allowlist and schema check, then run. Never raises for bad model input."""
    t0 = time.perf_counter()
    if name not in TOOLS:
        return Call(name, args, False, error=f"tool '{name}' is not on the allowlist")
    tool = TOOLS[name]
    try:
        parsed = tool.args.model_validate(args)
    except ValidationError as e:
        return Call(name, args, False, error=f"invalid arguments: {e.errors()[0]['msg']}")
    result = tool.fn(parsed)
    return Call(name, args, True, result=result, seconds=time.perf_counter() - t0)


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs so the grammar converter sees a self-contained schema."""
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def arg_schema(name: str) -> dict[str, Any]:
    s = _inline(TOOLS[name].args.model_json_schema())
    s["required"] = list(s.get("properties", {}))
    return s
