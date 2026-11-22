"""Module 2: truth recovery (interval coverage at about the nominal rate) and analytic checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from kapitalkompass.bayes.pd_model import GH_W, GH_X, segment_ttc_pd
from kapitalkompass.bayes.run import clopper_pearson_upper, recovery_coverage


def test_vasicek_year_pd_averages_to_ttc() -> None:
    """E_z[N((N^-1(PD) - sqrt(rho) z)/sqrt(1-rho))] = PD: the generator's TTC truth is exact."""
    pd, rho = 0.005, 0.08
    m, _ = quad(
        lambda z: norm.cdf((norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1 - rho)) * norm.pdf(z),
        -12,
        12,
    )
    assert m == pytest.approx(pd, rel=1e-8)


def test_gauss_hermite_ttc() -> None:
    post = {
        "a": np.zeros((1, 3)),
        "b": np.zeros((1, 4)),
        "tau": np.zeros(1),
        "u_raw": np.zeros((1, 12)),
        "sigma_t": np.zeros(1),
    }
    assert segment_ttc_pd(post, 3, 4) == pytest.approx(np.full((1, 12), 0.5))
    assert np.sum(GH_W * GH_X**2) == pytest.approx(1.0)


def test_clopper_pearson_zero_defaults_closed_form() -> None:
    n = np.array([500])
    assert clopper_pearson_upper(np.array([0]), n, 0.95)[0] == pytest.approx(
        1 - 0.05 ** (1 / 500), rel=1e-10
    )


@pytest.mark.slow
def test_truth_recovery_coverage() -> None:
    """Known PDs in: credible intervals cover the truth at about the nominal rate.
    Segments within one replication share the simulated cycle, so coverage is clustered;
    16 replications x 12 segments with bands wide enough for that clustering."""
    cov = recovery_coverage(n_reps=16, levels=(0.5, 0.9), seed0=1000)
    assert 0.35 <= cov[0.5] <= 0.70, cov
    assert 0.78 <= cov[0.9] <= 0.99, cov
