"""Multi-factor Gaussian / t copula credit portfolio model, simulated in PyTorch.

Obligor i in sector s defaults when
    sqrt(rho_s) Z_s + sqrt(1 - rho_s) eps_i  <  c_i
with Z_s = sqrt(w) G + sqrt(1 - w) S_s (sector factors with pairwise correlation w).
Gaussian: c_i = N^-1(PD_i). t copula: the latent variable is divided by sqrt(W / nu),
W ~ chi2(nu), and c_i = t_nu^-1(PD_i), so marginal PDs are preserved.

Scenarios are simulated in chunks; each chunk uses its own generator seeded with
(seed, chunk index), so results do not depend on chunk scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.stats import t as student_t


@dataclass
class Portfolio:
    pd: NDArray[np.float64]
    ead: NDArray[np.float64]
    lgd: NDArray[np.float64]
    sector: NDArray[np.int64]  # 0..n_sectors-1
    group: NDArray[np.int64]  # reporting group (segment) 0..n_groups-1
    rho_sector: NDArray[np.float64]  # per sector
    global_weight: float
    group_names: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.pd)

    @property
    def n_groups(self) -> int:
        return int(self.group.max()) + 1


@dataclass
class SimResult:
    loss: NDArray[np.float64]  # (S,) total loss
    group_loss: NDArray[np.float32]  # (S, G) loss per group
    scenarios: int
    device: str
    copula: str


def _thresholds(pd: NDArray[np.float64], copula: str, nu: float) -> NDArray[np.float64]:
    return student_t.ppf(pd, nu) if copula == "t" else norm.ppf(pd)


def simulate(
    pf: Portfolio,
    scenarios: int,
    seed: int,
    device: str = "cuda",
    chunk: int = 16384,
    block: int = 12000,
    copula: str = "gaussian",
    nu: float = 6.0,
) -> SimResult:
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    # Sort obligors by sector so each block has one sector factor.
    order = np.argsort(pf.sector, kind="stable")
    sec = pf.sector[order]
    a_np = np.sqrt(pf.rho_sector[sec])
    b_np = np.sqrt(1.0 - pf.rho_sector[sec])
    c_np = _thresholds(pf.pd[order], copula, nu)
    w_np = np.zeros((pf.n, pf.n_groups), dtype=np.float32)
    w_np[np.arange(pf.n), pf.group[order]] = (pf.ead * pf.lgd)[order]

    blocks: list[tuple[int, slice]] = []
    for s in np.unique(sec):
        idx = np.flatnonzero(sec == s)
        for start in range(idx[0], idx[-1] + 1, block):
            blocks.append((int(s), slice(start, min(start + block, idx[-1] + 1))))

    a = torch.tensor(a_np, dtype=torch.float32, device=dev)
    b = torch.tensor(b_np, dtype=torch.float32, device=dev)
    c = torch.tensor(c_np, dtype=torch.float32, device=dev)
    w = torch.tensor(w_np, dtype=torch.float32, device=dev)
    n_sec = len(pf.rho_sector)
    sw, sw1 = math.sqrt(pf.global_weight), math.sqrt(1.0 - pf.global_weight)

    group_loss = np.empty((scenarios, pf.n_groups), dtype=np.float32)
    for ci, start in enumerate(range(0, scenarios, chunk)):
        s_n = min(chunk, scenarios - start)
        gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + ci)
        g = torch.randn(s_n, 1, generator=gen, device=dev)
        z = sw * g + sw1 * torch.randn(s_n, n_sec, generator=gen, device=dev)
        if copula == "t":
            # W ~ chi2(nu) as a sum of nu squared normals from the same seeded generator.
            k = int(nu)
            if k != nu:
                raise ValueError("t copula needs integer nu")
            wch = (torch.randn(s_n, k, generator=gen, device=dev) ** 2).sum(1, keepdim=True)
            scale = torch.sqrt(wch / nu)
        acc = torch.zeros(s_n, pf.n_groups, dtype=torch.float32, device=dev)
        for s, sl in blocks:
            eps = torch.randn(s_n, sl.stop - sl.start, generator=gen, device=dev)
            # latent = a*Z_s + b*eps ; default if latent < c (Gaussian) or < c*scale (t)
            thr = c[sl].unsqueeze(0) * scale if copula == "t" else c[sl].unsqueeze(0)
            thr = (thr - a[sl].unsqueeze(0) * z[:, s : s + 1]) / b[sl].unsqueeze(0)
            acc += (eps < thr).to(torch.float32) @ w[sl]
            del eps, thr
        group_loss[start : start + s_n] = acc.cpu().numpy()
    loss = group_loss.astype(np.float64).sum(axis=1)
    return SimResult(loss, group_loss, scenarios, str(dev), copula)


@dataclass
class RiskMeasures:
    el_analytic: float
    el_simulated: float
    var: float
    es: float
    group_es_contrib: NDArray[np.float64]
    group_var_contrib: NDArray[np.float64]
    group_el: NDArray[np.float64]


def risk_measures(pf: Portfolio, res: SimResult, alpha: float) -> RiskMeasures:
    """VaR/ES and Euler contributions. ES contributions E[L_g | L >= VaR] sum to ES exactly
    (tail = the ceil((1-alpha) S) largest scenarios)."""
    loss = res.loss
    k = math.ceil((1.0 - alpha) * len(loss))
    tail = np.argpartition(loss, len(loss) - k)[len(loss) - k :]
    var = float(np.min(loss[tail]))
    es = float(np.mean(loss[tail]))
    gl = res.group_loss.astype(np.float64)
    es_c = gl[tail].mean(axis=0)
    # VaR contributions: average over a window of scenarios ranked around the quantile.
    ranks = np.argsort(loss)
    pos = len(loss) - k
    h = max(10, k // 10)
    win = ranks[max(0, pos - h) : pos + h]
    var_c = gl[win].mean(axis=0)
    var_c *= var / var_c.sum()
    el_g = np.bincount(pf.group, weights=pf.pd * pf.lgd * pf.ead, minlength=pf.n_groups)
    return RiskMeasures(
        el_analytic=float(el_g.sum()),
        el_simulated=float(loss.mean()),
        var=var,
        es=es,
        group_es_contrib=es_c,
        group_var_contrib=var_c,
        group_el=el_g,
    )


def hhi(x: NDArray[np.float64]) -> float:
    s = x / x.sum()
    return float(np.sum(s * s))
