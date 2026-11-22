"""Module 1 pipeline step: loss distribution, VaR/ES, concentration, Euler allocation."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import polars as pl
import torch

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.credit.asrf import asrf_k, irb_k
from kapitalkompass.credit.portfolio import Portfolio, hhi, risk_measures, simulate
from kapitalkompass.data.schemas import SECTORS, SEGMENTS, LoanBook
from kapitalkompass.results import write_json, write_table


def build_portfolio(lb: pl.DataFrame, cfg: dict[str, Any]) -> Portfolio:
    sec_idx = {s: i for i, s in enumerate(SECTORS)}
    seg_idx = {s: i for i, s in enumerate(SEGMENTS)}
    return Portfolio(
        pd=lb["pd"].to_numpy(),
        ead=lb["ead_eur"].to_numpy(),
        lgd=lb["lgd"].to_numpy(),
        sector=lb["sector"].replace_strict(sec_idx, return_dtype=pl.Int64).to_numpy(),
        group=lb["segment"].replace_strict(seg_idx, return_dtype=pl.Int64).to_numpy(),
        rho_sector=np.array([cfg["sector_rho"][s] for s in SECTORS]),
        global_weight=float(cfg["global_weight"]),
        group_names=list(SEGMENTS),
    )


def resolve_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(device: str | None = None, scenarios: int | None = None) -> dict[str, Any]:
    mcfg = load_config("model")
    cfg = mcfg["credit"]
    dev = resolve_device(device)
    lb = LoanBook.validate(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"))
    pf = build_portfolio(lb, cfg)
    alpha = float(cfg["confidence"])
    n_sim = int(scenarios or cfg["scenarios"])
    seed = int(mcfg["seed"])

    t0 = time.perf_counter()
    res = simulate(pf, n_sim, seed, dev, cfg["chunk_scenarios"], cfg["obligor_block"])
    t_main = time.perf_counter() - t0
    rm = risk_measures(pf, res, alpha)

    n_t = min(n_sim, int(cfg["t_scenarios"]))
    res_t = simulate(
        pf, n_t, seed + 1, dev, cfg["chunk_scenarios"], cfg["obligor_block"], "t", cfg["t_nu"]
    )
    rm_t = risk_measures(pf, res_t, alpha)

    # One-factor run (all sector factors = one global factor): isolates name concentration,
    # because the ASRF formula is exactly this model in the infinitely granular limit.
    pf_1f = dataclasses.replace(pf, global_weight=1.0)
    res_1f = simulate(pf_1f, n_t, seed + 2, dev, cfg["chunk_scenarios"], cfg["obligor_block"])
    rm_1f = risk_measures(pf_1f, res_1f, alpha)

    # Analytic references
    rho_i = pf.rho_sector[pf.sector]
    asrf_ul = float(np.sum(pf.ead * asrf_k(pf.pd, pf.lgd, rho_i, alpha)))
    irb_capital = float(np.sum(pf.ead * irb_k(pf.pd, pf.lgd, lb["maturity_years"].clip(1, 5))))

    ead = pf.ead
    top10 = float(np.sort(ead)[-10:].sum() / ead.sum())
    sector_ead = np.bincount(pf.sector, weights=ead)
    ec = rm.group_es_contrib - rm.group_el

    seg = pl.DataFrame(
        {
            "segment": SEGMENTS,
            "ead_eur": np.bincount(pf.group, weights=ead, minlength=len(SEGMENTS)),
            "el_eur": rm.group_el,
            "es_contrib_eur": rm.group_es_contrib,
            "var_contrib_eur": rm.group_var_contrib,
            "economic_capital_eur": ec,
            "t_es_contrib_eur": rm_t.group_es_contrib,
        }
    ).with_columns(ec_share=pl.col("economic_capital_eur") / pl.col("economic_capital_eur").sum())
    write_table("credit/segment_capital.csv", seg)

    qs = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]
    dist = pl.DataFrame(
        {
            "quantile": qs,
            "loss_gaussian_eur": np.quantile(res.loss, qs),
            "loss_t_eur": np.quantile(res_t.loss, qs),
        }
    )
    write_table("credit/loss_quantiles.csv", dist)
    hist, edges = np.histogram(res.loss, bins=200)
    write_table(
        "credit/loss_histogram.csv",
        pl.DataFrame({"bin_left_eur": edges[:-1], "bin_right_eur": edges[1:], "count": hist}),
    )

    summary = {
        "n_obligors": pf.n,
        "n_sectors": len(SECTORS),
        "n_segments": len(SEGMENTS),
        "total_ead_eur": float(ead.sum()),
        "scenarios": n_sim,
        "confidence": alpha,
        "device": dev,
        "copula": "gaussian",
        "el_analytic_eur": rm.el_analytic,
        "el_simulated_eur": rm.el_simulated,
        "var_eur": rm.var,
        "es_eur": rm.es,
        "unexpected_loss_var_eur": rm.var - rm.el_analytic,
        "economic_capital_eur": rm.es - rm.el_analytic,
        "economic_capital_definition": "ES_99.9 - EL (Euler ES contributions minus segment EL)",
        "euler_es_sum_eur": float(rm.group_es_contrib.sum()),
        "asrf_ul_eur": asrf_ul,
        "sim_ul_over_asrf_ul": (rm.var - rm.el_analytic) / asrf_ul,
        "irb_capital_requirement_eur": irb_capital,
        "one_factor": {
            "scenarios": n_t,
            "var_eur": rm_1f.var,
            "es_eur": rm_1f.es,
            "ul_var_eur": rm_1f.var - rm_1f.el_analytic,
            "ul_over_asrf_ul": (rm_1f.var - rm_1f.el_analytic) / asrf_ul,
            "note": "one-factor UL above ASRF UL = name-concentration (granularity) add-on",
        },
        "multifactor_diversification": 1.0
        - (rm.var - rm.el_analytic) / (rm_1f.var - rm_1f.el_analytic),
        "t_copula": {
            "nu": cfg["t_nu"],
            "scenarios": n_t,
            "var_eur": rm_t.var,
            "es_eur": rm_t.es,
            "es_ratio_t_over_gaussian": rm_t.es / rm.es,
            "el_simulated_eur": rm_t.el_simulated,
        },
        "concentration": {
            "hhi_name": hhi(ead),
            "effective_number_names": 1.0 / hhi(ead),
            "top10_share": top10,
            "hhi_sector": hhi(sector_ead),
            "n_large_names": int(lb["is_large"].sum()),
            "large_names_share": float(lb.filter("is_large")["ead_eur"].sum()) / float(ead.sum()),
        },
    }
    write_json("credit/summary.json", summary)
    print(
        f"credit: {n_sim:,} scen on {dev} in {t_main:.1f}s; VaR {rm.var / 1e6:.1f}m, "
        f"ES {rm.es / 1e6:.1f}m, EL {rm.el_analytic / 1e6:.1f}m, sim/ASRF UL "
        f"{summary['sim_ul_over_asrf_ul']:.3f}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--scenarios", type=int, default=None)
    a = ap.parse_args()
    run(a.device, a.scenarios)
