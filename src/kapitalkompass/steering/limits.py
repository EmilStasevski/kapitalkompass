"""Limit system: traffic lights on limit utilisation, escalation rule, IRRBB hedge action."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import linprog

from kapitalkompass.alm.curves import SCENARIOS, ZeroCurve, shock_vector
from kapitalkompass.alm.irrbb import Position, bullet, delta_nii, eve, floating


def light(utilisation: float, amber_from: float, red_from: float) -> str:
    """Red = limit breached (utilisation strictly above red_from); amber from amber_from."""
    if utilisation > red_from + 1e-12:
        return "red"
    if utilisation >= amber_from:
        return "amber"
    return "green"


def lcr_light(ratio: float, green_from: float, amber_from: float) -> str:
    if ratio >= green_from:
        return "green"
    if ratio >= amber_from:
        return "amber"
    return "red"


def limit_table(m: dict[str, float], cfg: dict[str, Any]) -> pl.DataFrame:
    """m: measured values. Utilisation = exposure / limit amount."""
    cover = m["risk_cover"]
    a, r = cfg["amber_from"], cfg["red_from"]
    it = cfg["items"]
    rows = []

    def add(name: str, exposure: float, limit_amt: float, unit: str) -> None:
        u = exposure / limit_amt
        rows.append(
            {
                "limit": name,
                "description": it[name]["description"],
                "exposure": exposure,
                "limit_value": limit_amt,
                "unit": unit,
                "utilisation": u,
                "status": light(u, a, r),
            }
        )

    add(
        "credit_risk_capital",
        m["credit_ec"],
        it["credit_risk_capital"]["limit_share_of_cover"] * cover,
        "EUR",
    )
    add("irrbb_eve", m["eve_loss"], it["irrbb_eve"]["limit_share_of_cover"] * m["tier1"], "EUR")
    add(
        "total_risk_cover",
        m["credit_ec"] + m["eve_loss"],
        it["total_risk_cover"]["limit_share_of_cover"] * cover,
        "EUR",
    )
    add("nii_shock", m["nii_loss"], it["nii_shock"]["limit_share_of_cover"] * m["tier1"], "EUR")
    add("sector_concentration", m["max_sector_share"], it["sector_concentration"]["limit"], "share")
    add("single_name", m["max_name"] / m["tier1"], it["single_name"]["limit"], "share of CET1")
    lc = cfg["lcr"]
    rows.append(
        {
            "limit": "lcr",
            "description": (
                f"LCR vs internal floor {lc['green_from']:.0%}; red below {lc['amber_from']:.0%}"
            ),
            "exposure": m["lcr"],
            "limit_value": lc["green_from"],
            "unit": "ratio",
            "utilisation": lc["green_from"] / m["lcr"],
            "status": lcr_light(m["lcr"], lc["green_from"], lc["amber_from"]),
        }
    )
    t = pl.DataFrame(rows)
    return t.with_columns(escalation=pl.col("status").replace_strict(cfg["escalation"]))


def swap_legs(curve: ZeroCurve, tenor: float) -> list[Position]:
    """1 EUR receive-fixed swap: fixed bullet leg (asset) minus floating leg (liability)."""
    return [
        bullet("swap_fixed", "asset", 1.0, tenor, float(curve.rate(tenor))),
        floating("swap_float", "liability", 1.0, 0.25, float(curve.rate(0.25))),
    ]


def swap_unit_deltas(
    curve: ZeroCurve, tenor: float, shocks_bp: dict[str, float], x: float, floor: float | str
) -> dict[str, dict[str, float]]:
    """dEVE and dNII (one-year horizon) of a 1 EUR receive-fixed swap per scenario."""
    legs = swap_legs(curve, tenor)
    e0 = eve(legs, curve)
    out = {}
    for sc in SCENARIOS:
        d = shock_vector(
            sc,
            curve.t,
            shocks_bp["parallel"] / 1e4,
            shocks_bp["short"] / 1e4,
            shocks_bp["long"] / 1e4,
            x,
        )
        shocked = curve.shifted(d, floor)
        out[sc] = {"eve": eve(legs, shocked) - e0, "nii": delta_nii(legs, curve, shocked, 1.0)}
    return out


def hedge_notional(
    bank_deve: dict[str, float], unit: dict[str, float], max_loss: float
) -> dict[str, Any]:
    """Smallest |h| with dEVE_bank(s) + h * unit(s) >= -max_loss for all six scenarios.
    h > 0 receive fixed, h < 0 pay fixed. Solved as an LP over h = h+ - h-."""
    a_ub, b_ub = [], []
    for sc in SCENARIOS:
        # -(bank + h unit) <= max_loss  ->  -unit*h <= max_loss + bank
        a_ub.append([-unit[sc], unit[sc]])
        b_ub.append(max_loss + bank_deve[sc])
    res = linprog(
        [1.0, 1.0],
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None), (0, None)],
        method="highs",
    )
    if not res.success:
        return {"feasible": False}
    h = float(res.x[0] - res.x[1])
    after = {sc: bank_deve[sc] + h * unit[sc] for sc in SCENARIOS}
    worst = min(after, key=lambda k: after[k])
    return {
        "feasible": True,
        "notional_eur": h,
        "direction": "receive_fixed" if h > 0 else "pay_fixed",
        "delta_eve_after": after,
        "worst_after_scenario": worst,
        "worst_after_eur": after[worst],
    }
