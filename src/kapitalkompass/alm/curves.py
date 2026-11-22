"""Zero curve interpolation and BCBS 368 Annex 2 shock scenarios."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

SCENARIOS = ["parallel_up", "parallel_down", "steepener", "flattener", "short_up", "short_down"]


class ZeroCurve:
    """Continuously compounded zero rates, linear in tenor, flat outside the grid."""

    def __init__(self, tenors: ArrayLike, rates: ArrayLike) -> None:
        self.t = np.asarray(tenors, dtype=float)
        self.r = np.asarray(rates, dtype=float)

    def rate(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.interp(np.asarray(t, dtype=float), self.t, self.r)

    def df(self, t: ArrayLike) -> NDArray[np.float64]:
        tt = np.asarray(t, dtype=float)
        return np.exp(-self.rate(tt) * tt)

    def shifted(self, delta: NDArray[np.float64], floor: float | None = None) -> ZeroCurve:
        r = self.r + delta
        if floor is not None:
            r = np.maximum(r, np.minimum(floor, self.r))  # floor never raises the base rate
        return ZeroCurve(self.t, r)


def shock_vector(
    scenario: str, t: ArrayLike, parallel: float, short: float, long: float, x: float = 4.0
) -> NDArray[np.float64]:
    """Rate change (decimal) at tenors t. Shock sizes in decimals.
    BCBS 368 Annex 2: S_short(t) = exp(-t/x), S_long = 1 - S_short,
    steepener = -0.65|dR_short| + 0.9|dR_long|, flattener = +0.8|dR_short| - 0.6|dR_long|."""
    tt = np.asarray(t, dtype=float)
    s_short = np.exp(-tt / x)
    d_short = short * s_short
    d_long = long * (1.0 - s_short)
    out = {
        "parallel_up": np.full_like(tt, parallel),
        "parallel_down": np.full_like(tt, -parallel),
        "short_up": d_short,
        "short_down": -d_short,
        "steepener": -0.65 * np.abs(d_short) + 0.9 * np.abs(d_long),
        "flattener": 0.8 * np.abs(d_short) - 0.6 * np.abs(d_long),
        "none": np.zeros_like(tt),
    }
    return out[scenario]
