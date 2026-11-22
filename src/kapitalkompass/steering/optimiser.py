"""Constrained re-allocation of segment volumes (cvxpy LP).

Decision: volume multiplier x_s per segment. Economic capital is linearised with the Euler
contributions (exact first-order sensitivity of a degree-1 homogeneous risk measure);
the optimised book is re-simulated ex post to check the capital budget.
Loan volume changes are funded from / released into Level 1 cash, so
HQLA = HQLA_0 - sum_s (x_s - 1) EAD_s and RSF = RSF_0 + sum_s (x_s - 1) EAD_s rsf_s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray


@dataclass
class Problem:
    names: list[str]
    profit: NDArray[np.float64]  # risk-adjusted profit per segment (EUR)
    capital: NDArray[np.float64]  # Euler EC per segment (EUR)
    ead: NDArray[np.float64]
    sector: NDArray[np.int64]
    sector_names: list[str]
    sector_max: NDArray[np.float64]
    hurdle: float
    capital_budget: float
    lo: float
    hi: float
    # liquidity: HQLA0 - dEAD >= floor * net_out ; RSF0 + rsf . dEAD <= ASF / floor
    hqla0: float | None = None
    net_out: float | None = None
    lcr_floor: float | None = None
    rsf0: float | None = None
    asf: float | None = None
    rsf_w: NDArray[np.float64] | None = None
    nsfr_floor: float | None = None


@dataclass
class Solution:
    x: NDArray[np.float64]
    objective: float
    status: str
    constraints: dict[str, dict[str, Any]]


def solve(p: Problem) -> Solution:
    n = len(p.names)
    x = cp.Variable(n)
    eva = p.profit - p.hurdle * p.capital
    cons: dict[str, Any] = {
        "capital_budget": p.capital @ x <= p.capital_budget,
        "growth_lower": x >= p.lo,
        "growth_upper": x <= p.hi,
    }
    total = p.ead @ x
    for k, s in enumerate(p.sector_names):
        mask = (p.sector == k).astype(float)
        cons[f"sector_share_{s}"] = (mask * p.ead) @ x <= p.sector_max[k] * total
    d_ead = p.ead @ (x - 1.0)
    if p.lcr_floor is not None:
        assert p.hqla0 is not None and p.net_out is not None
        cons["lcr_floor"] = p.hqla0 - d_ead >= p.lcr_floor * p.net_out
    if p.nsfr_floor is not None:
        assert p.rsf0 is not None and p.asf is not None and p.rsf_w is not None
        cons["nsfr_floor"] = p.rsf0 + (p.rsf_w * p.ead) @ (x - 1.0) <= p.asf / p.nsfr_floor
    prob = cp.Problem(cp.Maximize(eva @ x), list(cons.values()))
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        return Solution(np.full(n, np.nan), float("nan"), str(prob.status), {})
    xv = np.asarray(x.value, dtype=float)
    info: dict[str, dict[str, Any]] = {}
    for name, c in cons.items():
        dual = np.atleast_1d(np.asarray(c.dual_value, dtype=float))
        # slack >= 0 for a satisfied constraint (cvxpy: residual of expr <= 0 form)
        viol = np.atleast_1d(np.asarray(c.violation(), dtype=float))
        info[name] = {
            "shadow_price": dual.tolist() if dual.size > 1 else float(dual[0]),
            "binding": bool(np.any(np.abs(dual) > 1e-6)),
            "max_violation": float(viol.max()),
        }
    return Solution(xv, float(prob.value), str(prob.status), info)


def check_feasible(p: Problem, x: NDArray[np.float64], tol: float = 1e-6) -> list[str]:
    """Independent feasibility check (no cvxpy), relative tolerance."""
    bad = []
    if p.capital @ x > p.capital_budget * (1 + tol):
        bad.append("capital_budget")
    if np.any(x < p.lo - tol) or np.any(x > p.hi + tol):
        bad.append("growth_bounds")
    total = p.ead @ x
    for k, s in enumerate(p.sector_names):
        if (p.ead * (p.sector == k)) @ x > p.sector_max[k] * total * (1 + tol):
            bad.append(f"sector_{s}")
    d = p.ead @ (x - 1)
    if (
        p.lcr_floor is not None
        and p.hqla0 is not None
        and p.net_out is not None
        and (p.hqla0 - d) < p.lcr_floor * p.net_out * (1 - tol)
    ):
        bad.append("lcr")
    if (
        p.nsfr_floor is not None
        and p.rsf0 is not None
        and p.asf is not None
        and p.rsf_w is not None
        and p.rsf0 + (p.rsf_w * p.ead) @ (x - 1) > p.asf / p.nsfr_floor * (1 + tol)
    ):
        bad.append("nsfr")
    return bad
