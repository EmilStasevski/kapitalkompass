"""Hierarchical Bayesian PD calibration for low-default segments (NumPyro).

logit p_{s,g,t} = a_s + b_g + u_{s,g} + sigma_t * z_t
  a_s   ~ N(logit(prior PD of sector s), prior_sd)   sector level (average grade)
  b_g   ordered grade offsets, centred to mean zero (positive HalfNormal increments)
  u_sg  ~ N(0, tau), tau ~ HalfNormal(0.5)           segment deviation (shrinks to a_s + b_g)
  z_t   ~ N(0, 1), sigma_t ~ HalfNormal(0.5)         common year (cycle) effect
Through-the-cycle PD = E_z[invlogit(a_s + b_g + u_sg + sigma_t z)] by Gauss-Hermite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS
from scipy.special import logit

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(30)
GH_W = GH_W / GH_W.sum()


@dataclass
class PDData:
    sector: NDArray[np.int64]  # per observation (segment-year)
    grade: NDArray[np.int64]  # 0..G-1
    seg: NDArray[np.int64]  # 0..S*G-1
    year: NDArray[np.int64]  # 0..T-1
    n: NDArray[np.int64]
    d: NDArray[np.int64]
    n_sectors: int
    n_grades: int
    n_years: int


def model(data: PDData, prior_logit: NDArray[np.float64], prior_sd: float) -> None:
    a = numpyro.sample("a", dist.Normal(jnp.asarray(prior_logit), prior_sd))
    b1 = numpyro.sample("b1", dist.Normal(0.0, 2.0))
    inc = numpyro.sample("inc", dist.HalfNormal(1.5).expand([data.n_grades - 1]))
    b_raw = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(inc)]) + b1
    b = numpyro.deterministic("b", b_raw - b_raw.mean())
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    u_raw = numpyro.sample("u_raw", dist.Normal(0, 1).expand([data.n_sectors * data.n_grades]))
    u = tau * u_raw
    sig_t = numpyro.sample("sigma_t", dist.HalfNormal(0.5))
    z = numpyro.sample("z", dist.Normal(0, 1).expand([data.n_years]))
    eta = a[data.sector] + b[data.grade] + u[data.seg] + sig_t * z[data.year]
    numpyro.sample(
        "d", dist.Binomial(total_count=jnp.asarray(data.n), logits=eta), obs=jnp.asarray(data.d)
    )


def fit(
    data: PDData,
    prior_logit: NDArray[np.float64],
    prior_sd: float,
    seed: int,
    warmup: int = 1000,
    samples: int = 1000,
    chains: int = 2,
) -> dict[str, NDArray[np.float64]]:
    numpyro.set_host_device_count(max(chains, 1))
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data, prior_logit, prior_sd)
    post = {k: np.asarray(v, dtype=np.float64) for k, v in mcmc.get_samples().items()}
    extra = mcmc.get_extra_fields()
    post["_divergences"] = np.asarray(extra["diverging"]).astype(np.float64)
    return post


def segment_ttc_pd(
    post: dict[str, NDArray[np.float64]], n_sectors: int, n_grades: int
) -> NDArray[np.float64]:
    """(draws, S*G) through-the-cycle PDs, segment index = s * G + g."""
    s_idx = np.repeat(np.arange(n_sectors), n_grades)
    g_idx = np.tile(np.arange(n_grades), n_sectors)
    eta = post["a"][:, s_idx] + post["b"][:, g_idx] + post["tau"][:, None] * post["u_raw"]
    x = eta[..., None] + post["sigma_t"][:, None, None] * GH_X[None, None, :]
    out: NDArray[np.float64] = (1.0 / (1.0 + np.exp(-x)) * GH_W).sum(-1)
    return out


def posterior_predictive_defaults(
    post: dict[str, NDArray[np.float64]], data: PDData, seed: int
) -> NDArray[np.int64]:
    """Replicated default counts (draws, obs) at the fitted year effects."""
    rng = np.random.default_rng(seed)
    eta = (
        post["a"][:, data.sector]
        + post["b"][:, data.grade]
        + post["tau"][:, None] * post["u_raw"][:, data.seg]
        + post["sigma_t"][:, None] * post["z"][:, data.year]
    )
    p = 1.0 / (1.0 + np.exp(-eta))
    return rng.binomial(data.n[None, :], p)


def prior_logits(sector_prior_pd: dict[str, float], sectors: list[str]) -> NDArray[np.float64]:
    return np.asarray(logit([sector_prior_pd[s] for s in sectors]), dtype=np.float64)


def summarise(draws: NDArray[np.float64], credible: float) -> dict[str, Any]:
    lo, hi = (1 - credible) / 2, 1 - (1 - credible) / 2
    return {
        "mean": draws.mean(0),
        "median": np.median(draws, 0),
        "lower": np.quantile(draws, lo, axis=0),
        "upper": np.quantile(draws, hi, axis=0),
    }
