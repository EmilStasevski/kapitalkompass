"""Experiments for the pre-registered hypotheses H1 (granularity) and H2 (shrinkage); see
config/hypotheses.yaml. Writes results/mlops/hypotheses_experiments.json.

    uv run python -m kapitalkompass.mlops.experiments
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from kapitalkompass.bayes.pd_model import fit, prior_logits, segment_ttc_pd, summarise
from kapitalkompass.bayes.run import to_pddata
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.credit.asrf import asrf_k
from kapitalkompass.credit.portfolio import hhi, risk_measures, simulate
from kapitalkompass.credit.run import build_portfolio, resolve_device
from kapitalkompass.data.history import generate_history
from kapitalkompass.data.schemas import GRADES, SECTORS, LoanBook
from kapitalkompass.results import write_json, write_table


def granularity(device: str | None = None) -> dict[str, Any]:
    ecfg = load_config("mlops")["experiments"]["granularity"]
    mcfg = load_config("model")
    cfg = mcfg["credit"]
    dev = resolve_device(device)
    lb = LoanBook.validate(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"))
    base = dataclasses.replace(build_portfolio(lb, cfg), global_weight=1.0)
    large = lb["is_large"].to_numpy()
    alpha = float(cfg["confidence"])
    rows = []
    for f in ecfg["large_scale"]:
        pf = dataclasses.replace(base, ead=np.where(large, base.ead * float(f), base.ead))
        res = simulate(
            pf,
            int(ecfg["scenarios"]),
            int(mcfg["seed"]) + 2,
            dev,
            cfg["chunk_scenarios"],
            cfg["obligor_block"],
        )
        rm = risk_measures(pf, res, alpha)
        asrf_ul = float(np.sum(pf.ead * asrf_k(pf.pd, pf.lgd, pf.rho_sector[pf.sector], alpha)))
        rows.append(
            {
                "large_scale": float(f),
                "hhi_name": hhi(pf.ead),
                "large_names_share": float(pf.ead[large].sum() / pf.ead.sum()),
                "ul_sim_eur": rm.var - rm.el_analytic,
                "ul_asrf_eur": asrf_ul,
                "addon": (rm.var - rm.el_analytic) / asrf_ul - 1.0,
            }
        )
    tab = pl.DataFrame(rows)
    write_table("mlops/granularity.csv", tab)
    hi = tab.filter(pl.col("hhi_name") >= 0.003)["addon"]
    lo = tab.filter(pl.col("hhi_name") <= 0.0015)["addon"]
    return {
        "scenarios": int(ecfg["scenarios"]),
        "device": dev,
        "portfolios": rows,
        "n_hhi_ge_0_003": hi.len(),
        "n_hhi_le_0_0015": lo.len(),
        "min_addon_hhi_ge_0_003": float(hi.min()) if hi.len() else None,  # type: ignore[arg-type]
        "max_addon_hhi_le_0_0015": float(lo.max()) if lo.len() else None,  # type: ignore[arg-type]
        "spearman_hhi_addon": float(spearmanr(tab["hhi_name"], tab["addon"]).statistic),
    }


def shrinkage() -> dict[str, Any]:
    ecfg = load_config("mlops")["experiments"]["shrinkage"]
    cfg = copy.deepcopy(load_config("bank"))
    cfg["default_history"]["years"] = [
        cfg["default_history"]["years"][0],
        cfg["default_history"]["train_until"],
    ]
    bcfg = load_config("model")["bayes"]
    truth = np.tile([cfg["loan_book"]["grades"][str(g)]["pd"] for g in GRADES], len(SECTORS))
    rows = []
    for rep in range(int(ecfg["replications"])):
        seed = int(ecfg["seed0"]) + rep
        h = generate_history(cfg, seed=seed)
        data = to_pddata(h, sorted(h["year"].unique().to_list()))
        post = fit(
            data,
            prior_logits(bcfg["sector_prior_pd"], SECTORS),
            float(bcfg["sector_prior_sd"]),
            seed=seed,
            warmup=int(ecfg["warmup"]),
            samples=int(ecfg["samples"]),
            chains=1,
        )
        bayes = summarise(segment_ttc_pd(post, len(SECTORS), len(GRADES)), 0.9)["mean"]
        d = np.bincount(data.seg, weights=data.d, minlength=len(truth))
        n = np.bincount(data.seg, weights=data.n, minlength=len(truth))
        naive = d / n
        low = d <= 3
        if not low.any():
            continue
        mae_b = float(np.mean(np.abs(bayes[low] - truth[low])) * 1e4)
        mae_n = float(np.mean(np.abs(naive[low] - truth[low])) * 1e4)
        rows.append(
            {
                "seed": seed,
                "low_default_segments": int(low.sum()),
                "mae_bp_bayes": mae_b,
                "mae_bp_naive": mae_n,
                "ratio": mae_b / mae_n,
                "bayes_better": mae_b < mae_n,
            }
        )
    tab = pl.DataFrame(rows)
    write_table("mlops/shrinkage.csv", tab)
    return {
        "replications_run": int(ecfg["replications"]),
        "replications_with_low_default": tab.height,
        "bayes_better_share": float(np.mean(tab["bayes_better"].to_numpy())),
        "median_mae_ratio": float(tab["ratio"].median() or 0.0),  # type: ignore[arg-type]
        "mean_mae_bp_bayes": float(tab["mae_bp_bayes"].mean() or 0.0),  # type: ignore[arg-type]
        "mean_mae_bp_naive": float(tab["mae_bp_naive"].mean() or 0.0),  # type: ignore[arg-type]
    }


def run(device: str | None = None) -> dict[str, Any]:
    out = {"granularity": granularity(device), "shrinkage": shrinkage()}
    write_json("mlops/hypotheses_experiments.json", out)
    g, s = out["granularity"], out["shrinkage"]
    print(
        f"experiments: granularity spearman {g['spearman_hhi_addon']:.2f}, "
        f"min add-on (HHI>=0.003) {g['min_addon_hhi_ge_0_003']}; shrinkage better share "
        f"{s['bayes_better_share']:.2f}, median ratio {s['median_mae_ratio']:.2f}"
    )
    return out


if __name__ == "__main__":
    run()
