"""Analytic ASRF / Basel IRB formulas (BCBS CRE31.5, checked in docs/versions.md)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm


def basel_corporate_rho(pd: ArrayLike) -> NDArray[np.float64]:
    """CRE31.5 corporate asset correlation R(PD)."""
    p = np.asarray(pd, dtype=float)
    w = (1.0 - np.exp(-50.0 * p)) / (1.0 - np.exp(-50.0))
    return 0.12 * w + 0.24 * (1.0 - w)


def maturity_adjustment_b(pd: ArrayLike) -> NDArray[np.float64]:
    """CRE31.5 b = (0.11852 - 0.05478 ln PD)^2."""
    return (0.11852 - 0.05478 * np.log(np.asarray(pd, dtype=float))) ** 2


def asrf_conditional_pd(pd: ArrayLike, rho: ArrayLike, q: float = 0.999) -> NDArray[np.float64]:
    """PD conditional on the systematic factor at its q-quantile (worst case)."""
    p, r = np.asarray(pd, dtype=float), np.asarray(rho, dtype=float)
    return norm.cdf((norm.ppf(p) + np.sqrt(r) * norm.ppf(q)) / np.sqrt(1.0 - r))


def asrf_k(pd: ArrayLike, lgd: ArrayLike, rho: ArrayLike, q: float = 0.999) -> NDArray[np.float64]:
    """Unexpected-loss capital per unit EAD, no maturity adjustment:
    K = LGD * [N((G(PD) + sqrt(R) G(q)) / sqrt(1 - R)) - PD]."""
    p, lg = np.asarray(pd, dtype=float), np.asarray(lgd, dtype=float)
    return lg * (asrf_conditional_pd(p, rho, q) - p)


def irb_k(pd: ArrayLike, lgd: ArrayLike, m: ArrayLike, q: float = 0.999) -> NDArray[np.float64]:
    """Full CRE31.5 corporate capital requirement K incl. maturity adjustment."""
    p = np.asarray(pd, dtype=float)
    r = basel_corporate_rho(p)
    b = maturity_adjustment_b(p)
    mm = np.asarray(m, dtype=float)
    return asrf_k(p, lgd, r, q) * (1.0 + (mm - 2.5) * b) / (1.0 - 1.5 * b)


def vasicek_loss_cdf(x: ArrayLike, pd: float, rho: float, lgd: float = 1.0) -> NDArray[np.float64]:
    """Large-portfolio (Vasicek) distribution of the loss fraction L/EAD."""
    u = np.clip(np.asarray(x, dtype=float) / lgd, 1e-300, 1 - 1e-16)
    return norm.cdf((np.sqrt(1.0 - rho) * norm.ppf(u) - norm.ppf(pd)) / np.sqrt(rho))


def vasicek_loss_quantile(
    q: ArrayLike, pd: float, rho: float, lgd: float = 1.0
) -> NDArray[np.float64]:
    qq = np.asarray(q, dtype=float)
    return lgd * norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(qq)) / np.sqrt(1.0 - rho))
