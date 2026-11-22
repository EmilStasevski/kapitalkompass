"""Module 2 pipeline step: Bayesian PD vs naive vs pooled, PPC, time split, conformal."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import polars as pl
from mapie.regression import CrossConformalRegressor
from scipy.stats import beta as beta_dist
from scipy.stats import binom
from sklearn.linear_model import Ridge

from kapitalkompass.bayes.pd_model import (
    PDData,
    fit,
    posterior_predictive_defaults,
    prior_logits,
    segment_ttc_pd,
    summarise,
)
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.history import DefaultHistory, generate_history
from kapitalkompass.data.schemas import GRADES, SECTORS, SEGMENTS
from kapitalkompass.results import write_json, write_table


def to_pddata(h: pl.DataFrame, years: list[int]) -> PDData:
    sec = {s: i for i, s in enumerate(SECTORS)}
    yr = {y: i for i, y in enumerate(years)}
    s = h["sector"].replace_strict(sec, return_dtype=pl.Int64).to_numpy()
    g = h["grade"].to_numpy() - 1
    return PDData(
        sector=s,
        grade=g,
        seg=s * len(GRADES) + g,
        year=h["year"].replace_strict(yr, return_dtype=pl.Int64).to_numpy(),
        n=h["obligors"].to_numpy(),
        d=h["defaults"].to_numpy(),
        n_sectors=len(SECTORS),
        n_grades=len(GRADES),
        n_years=len(years),
    )


def clopper_pearson_upper(d: np.ndarray, n: np.ndarray, conf: float) -> np.ndarray:
    """One-sided upper bound (zero defaults -> 1 - (1-conf)^(1/n), Pluto-Tasche style)."""
    return np.where(d >= n, 1.0, beta_dist.ppf(conf, d + 1, n - d))


def conformal(h: pl.DataFrame, train_until: int, conf: float, seed: int) -> dict[str, Any]:
    """Frequentist counterpart: cross-conformal (CV+) intervals for annual segment default
    rates from a ridge model on grade/sector dummies and the macro index; coverage on the
    held-out years."""

    def features(df: pl.DataFrame) -> np.ndarray:
        g = np.stack([(df["grade"] == k).to_numpy() for k in GRADES], 1).astype(float)
        s = np.stack([(df["sector"] == k).to_numpy() for k in SECTORS[1:]], 1).astype(float)
        m = df["macro"].to_numpy()[:, None]
        return np.hstack([g, s, m, g * m])

    tr, te = h.filter(pl.col("year") <= train_until), h.filter(pl.col("year") > train_until)
    y_tr = (tr["defaults"] / tr["obligors"]).to_numpy()
    y_te = (te["defaults"] / te["obligors"]).to_numpy()
    reg = CrossConformalRegressor(
        Ridge(alpha=1e-3), confidence_level=conf, method="plus", cv=10, random_state=seed
    )
    # Variance-stabilising arcsine-sqrt transform for binomial rates; it is monotone, so
    # back-transformed bounds keep the conformal coverage guarantee.
    reg.fit_conformalize(features(tr), np.arcsin(np.sqrt(y_tr)))
    _pred, iv = reg.predict_interval(features(te))
    back = np.sin(np.clip(iv[:, :, 0], 0.0, np.pi / 2)) ** 2
    lo, hi = back[:, 0], back[:, 1]
    covered = (y_te >= lo) & (y_te <= hi)
    per_grade = {str(g): float(covered[(te["grade"] == g).to_numpy()].mean()) for g in GRADES}
    return {
        "method": "MAPIE CrossConformalRegressor (CV+, 10 folds), Ridge on arcsin(sqrt(rate)), "
        "absolute score, bounds back-transformed",
        "confidence": conf,
        "n_test": len(y_te),
        "coverage": float(covered.mean()),
        "coverage_by_grade": per_grade,
        "mean_width": float(np.mean(hi - lo)),
        "width_by_grade": {
            str(g): float(np.mean((hi - lo)[(te["grade"] == g).to_numpy()])) for g in GRADES
        },
        "lower_bound_zero_share": float(np.mean(lo <= 0)),
    }


def recovery_coverage(
    n_reps: int, levels: tuple[float, ...], seed0: int, warmup: int = 300, samples: int = 300
) -> dict[float, float]:
    """Simulated truth recovery: refit on fresh histories with known TTC PDs and count how
    often each credible interval covers the truth."""
    cfg = copy.deepcopy(load_config("bank"))
    cfg["default_history"]["years"] = [
        cfg["default_history"]["years"][0],
        cfg["default_history"]["train_until"],
    ]
    bcfg = load_config("model")["bayes"]
    truth = np.tile([cfg["loan_book"]["grades"][str(g)]["pd"] for g in GRADES], len(SECTORS))
    hits = dict.fromkeys(levels, 0)
    for rep in range(n_reps):
        h = generate_history(cfg, seed=seed0 + rep)
        data = to_pddata(h, sorted(h["year"].unique().to_list()))
        post = fit(
            data,
            prior_logits(bcfg["sector_prior_pd"], SECTORS),
            float(bcfg["sector_prior_sd"]),
            seed=seed0 + rep,
            warmup=warmup,
            samples=samples,
            chains=1,
        )
        draws = segment_ttc_pd(post, len(SECTORS), len(GRADES))
        for lv in levels:
            s = summarise(draws, lv)
            hits[lv] += int(np.sum((truth >= s["lower"]) & (truth <= s["upper"])))
    return {lv: hits[lv] / (n_reps * len(truth)) for lv in levels}


def run_calibration(n_reps: int = 40) -> dict[str, Any]:
    levels = (0.5, 0.8, 0.9)
    cov = recovery_coverage(n_reps, levels, seed0=50_000)
    out = {
        "replications": n_reps,
        "intervals_per_level": n_reps * len(SEGMENTS),
        "coverage": {str(k): v for k, v in cov.items()},
    }
    write_json("bayes/calibration.json", out)
    print(f"bayes calibration: {out['coverage']}")
    return out


def run() -> dict[str, Any]:
    bcfg = load_config("bank")
    mcfg = load_config("model")
    cfg = mcfg["bayes"]
    seed = int(mcfg["seed"])
    split = int(bcfg["default_history"]["train_until"])
    h = DefaultHistory.validate(pl.read_parquet(SYNTH_DIR / "default_history.parquet"))
    tr = h.filter(pl.col("year") <= split)
    te = h.filter(pl.col("year") > split)
    years_tr = sorted(tr["year"].unique().to_list())

    data = to_pddata(tr, years_tr)
    post = fit(
        data,
        prior_logits(cfg["sector_prior_pd"], SECTORS),
        float(cfg["sector_prior_sd"]),
        seed,
        cfg["warmup"],
        cfg["samples"],
        cfg["chains"],
    )
    draws = segment_ttc_pd(post, len(SECTORS), len(GRADES))
    summ = summarise(draws, float(cfg["credible"]))

    agg = (
        tr.group_by("segment", "sector", "grade")
        .agg(pl.col("defaults").sum(), pl.col("obligors").sum(), pl.col("true_pd").first())
        .with_columns(
            seg_idx=pl.col("segment").replace_strict(
                {s: i for i, s in enumerate(SEGMENTS)}, return_dtype=pl.Int64
            )
        )
        .sort("seg_idx")
    )
    pooled = tr.group_by("grade").agg(pooled_pd=pl.col("defaults").sum() / pl.col("obligors").sum())
    agg = agg.join(pooled, on="grade").sort("seg_idx")
    d, n = agg["defaults"].to_numpy(), agg["obligors"].to_numpy()
    conf_one_sided = 1 - (1 - float(cfg["credible"])) / 2
    test_agg = te.group_by("segment").agg(
        test_defaults=pl.col("defaults").sum(), test_obligors=pl.col("obligors").sum()
    )
    tab = (
        agg.with_columns(
            naive_pd=pl.col("defaults") / pl.col("obligors"),
            bayes_mean=summ["mean"],
            bayes_lower=summ["lower"],
            bayes_upper=summ["upper"],
            conservative_pd=summ["upper"],
            cp_upper=clopper_pearson_upper(d, n, conf_one_sided),
        )
        .join(test_agg, on="segment")
        .sort("seg_idx")
        .drop("seg_idx")
    )

    truth = tab["true_pd"].to_numpy()
    low = (tab["defaults"] <= 3).to_numpy()

    def log_err(est: np.ndarray, mask: np.ndarray) -> float | None:
        e = est[mask]
        if np.any(e <= 0):
            return None  # undefined: zero PD estimate
        return float(np.mean(np.abs(np.log(e / truth[mask]))))

    def test_logscore(est: np.ndarray) -> float:
        ll = binom.logpmf(tab["test_defaults"].to_numpy(), tab["test_obligors"].to_numpy(), est)
        return float(np.sum(ll))

    estimators = {
        "naive": tab["naive_pd"].to_numpy(),
        "pooled_by_grade": tab["pooled_pd"].to_numpy(),
        "bayes_mean": tab["bayes_mean"].to_numpy(),
    }
    comparison = {
        k: {
            "mean_abs_log_error_low_default": log_err(v, low),
            "mean_abs_log_error_all": log_err(v, np.ones_like(low)),
            "mean_abs_error_bp_low_default": float(np.mean(np.abs(v[low] - truth[low])) * 1e4),
            "zero_pd_segments": int(np.sum(v <= 0)),
            "test_years_log_score": test_logscore(v),
        }
        for k, v in estimators.items()
    }
    covered = (truth >= tab["bayes_lower"].to_numpy()) & (truth <= tab["bayes_upper"].to_numpy())

    # Posterior predictive checks
    rep = posterior_predictive_defaults(post, data, seed + 5)
    obs_year = np.bincount(data.year, weights=data.d)
    rep_year = np.stack([np.bincount(data.year, weights=r, minlength=data.n_years) for r in rep])
    seg_obs = np.bincount(data.seg, weights=data.d)
    seg_rep = np.stack([np.bincount(data.seg, weights=r, minlength=len(SEGMENTS)) for r in rep])
    seg_lo, seg_hi = np.quantile(seg_rep, [0.05, 0.95], axis=0)
    ppc = {
        "p_value_var_annual_total": float(np.mean(rep_year.var(1) >= obs_year.var())),
        "p_value_max_segment_year": float(np.mean(rep.max(1) >= data.d.max())),
        "segments_inside_90pct_predictive": int(np.sum((seg_obs >= seg_lo) & (seg_obs <= seg_hi))),
        "n_segments": len(SEGMENTS),
    }
    write_table("bayes/segment_pd.csv", tab)

    summary = {
        "train_years": [years_tr[0], years_tr[-1]],
        "test_years": [split + 1, int(h["year"].to_numpy().max())],
        "credible": cfg["credible"],
        "low_default_segments": int(low.sum()),
        "zero_default_segments": int((tab["defaults"] == 0).sum()),
        "truth_covered": int(covered.sum()),
        "truth_covered_low_default": int(covered[low].sum()),
        "comparison": comparison,
        "ppc": ppc,
        "mcmc": {
            "chains": cfg["chains"],
            "warmup": cfg["warmup"],
            "samples": cfg["samples"],
            "divergences": int(post["_divergences"].sum()),
            "sigma_t_mean": float(post["sigma_t"].mean()),
            "tau_mean": float(post["tau"].mean()),
        },
        "prior": {
            "sector_prior_pd": cfg["sector_prior_pd"],
            "sector_prior_sd_logit": cfg["sector_prior_sd"],
            "source": "stated assumption (Destatis sector rates not retrievable, decision 16)",
        },
        "conformal": conformal(h, split, float(cfg["conformal_confidence"]), seed),
    }
    write_json("bayes/summary.json", summary)
    print(
        f"bayes: truth covered {covered.sum()}/{len(covered)}, low-default log err "
        f"bayes {comparison['bayes_mean']['mean_abs_log_error_low_default']:.3f} vs pooled "
        f"{comparison['pooled_by_grade']['mean_abs_log_error_low_default']:.3f}; "
        f"conformal coverage {summary['conformal']['coverage']:.2f}"
    )
    return summary


if __name__ == "__main__":
    run()
    run_calibration()
