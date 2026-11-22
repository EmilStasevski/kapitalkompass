"""IRRBB: repricing cash flows, EVE and NII under the six BCBS 368 shock scenarios.

EVE follows BCBS 368 para 101: rate-sensitive assets and liabilities, CET1 excluded,
non-rate-sensitive other assets/liabilities excluded. Coupons are set at the risk-free
rate of the position's tenor (commercial margins excluded, decision 21). NII: one-year
horizon, constant balance sheet, instantaneous shock that persists; positions reprice at
the shocked rate of their own tenor; deposits pass on a beta share of the short rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from kapitalkompass.alm.curves import SCENARIOS, ZeroCurve, shock_vector


@dataclass
class Position:
    name: str
    side: str  # asset | liability
    times: NDArray[np.float64]  # repricing cash-flow times (years)
    amounts: NDArray[np.float64]  # repricing cash flows (principal + coupons), EUR
    # NII repricing within the horizon: (time, notional, tenor of the new rate, beta)
    reprices: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "asset" else -1.0


def bullet(name: str, side: str, notional: float, maturity: float, coupon: float) -> Position:
    """Annual coupons, principal at maturity. Rolled over at maturity for NII."""
    ct = np.arange(1.0, np.floor(maturity) + 1.0)
    ct = ct[ct < maturity - 1e-9]
    times = np.append(ct, maturity)
    amounts = np.full(len(times), notional * coupon)
    amounts[-1] += notional
    rep = [(maturity, notional, maturity, 1.0)] if maturity < 1.0 else []
    return Position(name, side, times, amounts, rep)


def amortizing(name: str, side: str, notional: float, maturity: float, coupon: float) -> Position:
    """Linear monthly principal; interest on outstanding. Repaid principal within the
    horizon is re-lent at the shocked rate of the same tenor (constant balance sheet)."""
    n = max(1, int(np.ceil(maturity * 12 - 1e-9)))
    times = np.arange(1, n + 1) / 12.0
    princ = np.full(n, notional / n)
    outstanding = notional - princ * np.arange(n)
    amounts = princ + outstanding * coupon / 12.0
    rep = [
        (float(t), float(p), maturity, 1.0) for t, p in zip(times, princ, strict=True) if t < 1.0
    ]
    return Position(name, side, times, amounts, rep)


def floating(name: str, side: str, notional: float, reset: float, coupon: float) -> Position:
    """Repricing at the next reset (par at reset date, BCBS 368 para 106(2)). Resets are
    spread over the period, so the average first reset is reset/2 for NII."""
    return Position(
        name,
        side,
        np.array([reset]),
        np.array([notional * (1.0 + coupon * reset)]),
        [(reset / 2.0, notional, reset, 1.0)],
    )


def eve(positions: list[Position], curve: ZeroCurve) -> float:
    return float(sum(p.sign * np.sum(p.amounts * curve.df(p.times)) for p in positions))


def delta_nii(
    positions: list[Position], base: ZeroCurve, shocked: ZeroCurve, horizon: float
) -> float:
    total = 0.0
    for p in positions:
        for t, notional, tenor, beta in p.reprices:
            if t < horizon:
                dr = float(shocked.rate(tenor) - base.rate(tenor))
                total += p.sign * notional * beta * dr * (horizon - t)
    return total


def run_scenarios(
    positions: list[Position],
    base: ZeroCurve,
    shocks_bp: dict[str, float],
    x: float,
    floor: float | None,
    horizon: float,
) -> dict[str, dict[str, float]]:
    eve0 = eve(positions, base)
    out = {}
    for sc in [*SCENARIOS, "none"]:
        d = shock_vector(
            sc,
            base.t,
            shocks_bp["parallel"] / 1e4,
            shocks_bp["short"] / 1e4,
            shocks_bp["long"] / 1e4,
            x,
        )
        shocked = base.shifted(d, floor)
        out[sc] = {
            "delta_eve": eve(positions, shocked) - eve0,
            "delta_nii": delta_nii(positions, base, shocked, horizon),
        }
    return out
