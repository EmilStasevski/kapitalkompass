"""Seeded default history per segment and year (input for the Bayesian PD module).

Annual default counts are Binomial(n, p_t) with the Vasicek conditional PD
p_t = N((N^-1(PD) - sqrt(rho) z_t) / sqrt(1 - rho)), z_t ~ N(0, 1) shared across segments,
so E[p_t] equals the true through-the-cycle PD from the loan book. An observable macro
index m_t = loading * z_t + noise is published alongside.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandera.polars as pa
import polars as pl
from scipy.stats import norm

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.schemas import GRADES, SECTORS, SEGMENTS


class DefaultHistory(pa.DataFrameModel):
    year: int
    segment: str = pa.Field(isin=SEGMENTS)
    sector: str = pa.Field(isin=SECTORS)
    grade: int = pa.Field(isin=GRADES)
    obligors: int = pa.Field(gt=0)
    defaults: int = pa.Field(ge=0)
    macro: float
    true_pd: float = pa.Field(gt=0, lt=1)

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def defaults_le_obligors(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("defaults") <= pl.col("obligors"))


def generate_history(cfg: dict[str, Any], seed: int) -> pl.DataFrame:
    h = cfg["default_history"]
    grades_cfg = cfg["loan_book"]["grades"]
    rng = np.random.default_rng(seed + 101)
    years = np.arange(h["years"][0], h["years"][1] + 1)
    z = rng.standard_normal(len(years))
    macro = h["macro_loading"] * z + h["macro_noise_sd"] * rng.standard_normal(len(years))
    rho = h["cycle_rho"]
    rows = []
    for t, yr in enumerate(years):
        for s in SECTORS:
            for g in GRADES:
                pd_true = grades_cfg[str(g)]["pd"]
                n = round(h["cohort_size_by_grade"][str(g)] * h["sector_size_factor"][s])
                p_t = norm.cdf((norm.ppf(pd_true) - np.sqrt(rho) * z[t]) / np.sqrt(1 - rho))
                rows.append(
                    {
                        "year": int(yr),
                        "segment": f"{s}_{g}",
                        "sector": s,
                        "grade": g,
                        "obligors": n,
                        "defaults": int(rng.binomial(n, p_t)),
                        "macro": round(float(macro[t]), 6),
                        "true_pd": pd_true,
                    }
                )
    return DefaultHistory.validate(pl.DataFrame(rows))


def build(write: bool = True) -> pl.DataFrame:
    cfg = load_config("bank")
    df = generate_history(cfg, int(cfg["seed"]))
    if write:
        df.write_parquet(SYNTH_DIR / "default_history.parquet", statistics=False)
    return df
