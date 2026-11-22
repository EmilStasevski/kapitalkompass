"""LCR (BCBS LCR30/LCR40), NSFR (BCBS NSF30) and contractual/behavioural ladders."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def hqla_stock(l1: float, l2a: float, l2b: float) -> dict[str, float]:
    """Amounts after haircuts. LCR30.38/30.39 cap adjustments."""
    adj15 = max(l2b - 15.0 / 85.0 * (l1 + l2a), l2b - 15.0 / 60.0 * l1, 0.0)
    adj40 = max(l2a + l2b - adj15 - 2.0 / 3.0 * l1, 0.0)
    return {
        "l1": l1,
        "l2a": l2a,
        "l2b": l2b,
        "adj15": adj15,
        "adj40": adj40,
        "hqla": l1 + l2a + l2b - adj15 - adj40,
    }


def lcr(bs: dict[str, float], loan_inflows_30d: float, cfg: dict[str, Any]) -> dict[str, Any]:
    lv: dict[str, float] = {"L1": 0.0, "L2A": 0.0, "L2B": 0.0}
    for item, spec in cfg["hqla"].items():
        lv[spec["level"]] += bs.get(item, 0.0) * (1.0 - spec["haircut"])
    stock = hqla_stock(lv["L1"], lv["L2A"], lv["L2B"])
    out_items = {}
    for item, rate in cfg["outflow_rates"].items():
        amt = bs.get(item, 0.0)
        if item == "wholesale_unsecured_lt1y":
            amt *= cfg["wholesale_lt1y_maturing_30d_share"]
        out_items[item] = amt * rate
    outflows = sum(out_items.values())
    inflows_raw = loan_inflows_30d * cfg["inflow_rate_nonfinancial"]
    inflows = min(inflows_raw, cfg["inflow_cap"] * outflows)
    net = outflows - inflows
    return {
        **stock,
        "outflows": outflows,
        "outflows_by_item": out_items,
        "inflows_before_cap": inflows_raw,
        "inflows": inflows,
        "inflow_cap_binding": inflows_raw > cfg["inflow_cap"] * outflows,
        "net_outflows": net,
        "lcr": stock["hqla"] / net,
    }


def nsfr(
    bs: dict[str, float], loans_lt1y: float, loans_ge1y: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    asf_items = {k: bs.get(k, 0.0) * f for k, f in cfg["asf"].items()}
    amounts = dict(bs)
    amounts["loans_lt1y"] = loans_lt1y
    amounts["loans_ge1y"] = loans_ge1y
    rsf_items = {k: amounts.get(k, 0.0) * f for k, f in cfg["rsf"].items()}
    asf, rsf = sum(asf_items.values()), sum(rsf_items.values())
    return {
        "asf": asf,
        "rsf": rsf,
        "asf_by_item": asf_items,
        "rsf_by_item": rsf_items,
        "nsfr": asf / rsf,
    }


def ladder(
    flows: list[tuple[str, str, np.ndarray, np.ndarray]], edges: list[float]
) -> pl.DataFrame:
    """Bucket (name, side, times, amounts) cash flows; side asset (+) / liability (-)."""
    e = np.asarray(edges, dtype=float)
    labels = [f"{e[i]:g}-{e[i + 1]:g}y" for i in range(len(e) - 1)]
    assets = np.zeros(len(labels))
    liabs = np.zeros(len(labels))
    for _, side, t, a in flows:
        idx = np.clip(np.searchsorted(e, t, side="left") - 1, 0, len(labels) - 1)
        idx = np.where(t <= e[0], 0, idx)
        target = assets if side == "asset" else liabs
        np.add.at(target, idx, a)
    gap = assets - liabs
    return pl.DataFrame(
        {
            "bucket": labels,
            "inflows_eur": assets,
            "outflows_eur": liabs,
            "gap_eur": gap,
            "cumulative_gap_eur": np.cumsum(gap),
        }
    )
