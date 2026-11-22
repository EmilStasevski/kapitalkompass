"""Behavioural non-maturity deposits: simulated account histories, Weibull AFT survival
model (lifelines), core share and run-off profile capped per BCBS 368 Table 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter, WeibullAFTFitter


def simulate_accounts(cfg: dict[str, Any], seed: int) -> pl.DataFrame:
    """Accounts opened uniformly over the observation window; closure time Weibull with
    log-scale shifted by the balance effect; right-censored at the window end."""
    rng = np.random.default_rng(seed + 303)
    cats = list(cfg["categories"])
    share = np.array([cfg["categories"][c]["share"] for c in cats])
    n = int(cfg["n_accounts"])
    y0, y1 = cfg["observation_years"]
    window = float(y1 - y0 + 1)
    cat = rng.choice(len(cats), size=n, p=share)
    scale = np.array([cfg["categories"][c]["scale"] for c in cats])[cat]
    shape = np.array([cfg["categories"][c]["shape"] for c in cats])[cat]
    log_bal = rng.normal(0.0, 1.0, n)  # log(balance / category median)
    eff_scale = scale * np.exp(cfg["balance_effect"] * log_bal)
    life = eff_scale * rng.weibull(shape)
    opened = rng.uniform(0.0, window, n)
    exposure = window - opened
    observed = life <= exposure
    return pl.DataFrame(
        {
            "account_id": np.arange(n),
            "category": np.array(cats)[cat],
            "log_balance": log_bal,
            "balance_eur": np.round(np.exp(log_bal) * 20000.0, 2),
            "duration_years": np.where(observed, life, exposure),
            "closed": observed,
        }
    )


@dataclass
class DepositModel:
    aft: WeibullAFTFitter
    categories: list[str]

    def survival(self, category: str, t: np.ndarray, log_balance: np.ndarray) -> np.ndarray:
        """(len(t),) balance-weighted survival for a category (weights = balance)."""
        df = pd.DataFrame({"log_balance": log_balance})
        for c in self.categories[1:]:
            df[f"cat_{c}"] = 1.0 if c == category else 0.0
        sf = self.aft.predict_survival_function(df, times=t).to_numpy()  # (len(t), n)
        w = np.exp(log_balance)
        out: np.ndarray = (sf * w).sum(1) / w.sum()
        return out


def fit_model(accounts: pl.DataFrame, categories: list[str]) -> DepositModel:
    df = accounts.select("duration_years", "closed", "log_balance", "category").to_pandas()
    for c in categories[1:]:
        df[f"cat_{c}"] = (df["category"] == c).astype(float)
    df = df.drop(columns="category")
    df["closed"] = df["closed"].astype(int)
    aft = WeibullAFTFitter()
    # ancillary=True: the Weibull shape also depends on the covariates (category-specific shapes)
    aft.fit(df, duration_col="duration_years", event_col="closed", ancillary=True)
    return DepositModel(aft, categories)


def kaplan_meier(accounts: pl.DataFrame, category: str, t: np.ndarray) -> np.ndarray:
    sub = accounts.filter(pl.col("category") == category)
    km = KaplanMeierFitter().fit(sub["duration_years"].to_numpy(), sub["closed"].to_numpy())
    out: np.ndarray = km.survival_function_at_times(t).to_numpy()
    return out


@dataclass
class Profile:
    category: str
    stable_share: float  # model: balance surviving beyond the core horizon
    core_share: float  # after BCBS cap
    avg_maturity_core: float  # after BCBS cap (years)
    avg_maturity_uncapped: float
    times: np.ndarray  # cash-flow times (years) for the core part
    weights: np.ndarray  # share of the category balance flowing out at each time

    def runoff(self) -> tuple[np.ndarray, np.ndarray]:
        """Full behavioural profile: non-core overnight + core per weights."""
        t = np.concatenate([[1.0 / 365.0], self.times])
        w = np.concatenate([[1.0 - self.core_share], self.weights])
        return t, w


def behavioural_profile(
    model: DepositModel,
    category: str,
    log_balance: np.ndarray,
    cap: dict[str, float],
    horizon: float,
    max_years: float,
    grid_months: int = 240,
) -> Profile:
    t = np.arange(0, grid_months + 1) / 12.0
    t = t[t <= max_years]
    s = model.survival(category, t, log_balance)
    stable = float(np.interp(horizon, t, s))
    core = min(stable, cap["core_share"])
    # Core outflow density: run-off of the balance that survives beyond the horizon,
    # i.e. conditional exit times; mass still alive at max_years is placed at max_years.
    mask = t >= horizon
    tc, sc = t[mask], s[mask] / s[mask][0]
    dens = -np.diff(sc)
    times = 0.5 * (tc[1:] + tc[:-1])
    dens = np.append(dens, sc[-1])
    times = np.append(times, tc[-1])
    avg_unc = float(np.sum(dens * times))
    scale = min(1.0, cap["avg_maturity"] / avg_unc)
    times = times * scale  # compress the time axis so the average maturity meets the cap
    return Profile(
        category=category,
        stable_share=stable,
        core_share=core,
        avg_maturity_core=float(np.sum(dens * times)),
        avg_maturity_uncapped=avg_unc,
        times=times,
        weights=core * dens,
    )
