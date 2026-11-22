"""Funds transfer pricing: base curve plus a tenor-dependent liquidity premium."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from kapitalkompass.alm.curves import ZeroCurve


def liquidity_premium(t: ArrayLike, lp_long: float, tau: float) -> NDArray[np.float64]:
    tt = np.asarray(t, dtype=float)
    return lp_long * (1.0 - np.exp(-tt / tau))


def ftp_loans(
    lb: pl.DataFrame, curve: ZeroCurve, lp_long: float, tau: float, reset: float
) -> pl.DataFrame:
    """Per loan: funding tenor = average life (maturity / 2 for linear amortisation).
    Interest-rate component: base rate at the repricing tenor (average life for fixed
    loans, the reset tenor for floating loans) — matched with the client rate, so the
    rate risk sits in treasury. Liquidity component: LP at the average life for both.
    Client rate = base at repricing tenor + client margin."""
    life = lb["maturity_years"].to_numpy() / 2.0
    fixed = lb["fixed_rate"].to_numpy()
    reprice = np.where(fixed, life, reset)
    base = curve.rate(reprice)
    lp = liquidity_premium(life, lp_long, tau)
    ead = lb["ead_eur"].to_numpy()
    client = base + lb["client_margin"].to_numpy()
    ftp_rate = base + lp
    return lb.select("obligor_id", "segment", "ead_eur").with_columns(
        avg_life_years=life,
        base_rate=base,
        liquidity_premium=lp,
        ftp_rate=ftp_rate,
        client_rate=client,
        interest_income_eur=client * ead,
        ftp_charge_eur=ftp_rate * ead,
        nim_after_ftp_eur=(client - ftp_rate) * ead,
    )
