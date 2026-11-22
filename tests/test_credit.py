"""Module 1 analytic cross-checks: Monte Carlo vs ASRF / Vasicek, Euler additivity."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.credit.asrf import (
    asrf_k,
    basel_corporate_rho,
    irb_k,
    maturity_adjustment_b,
    vasicek_loss_cdf,
    vasicek_loss_quantile,
)
from kapitalkompass.credit.portfolio import Portfolio, risk_measures, simulate
from kapitalkompass.credit.run import build_portfolio

HAS_CUDA = torch.cuda.is_available()


def homogeneous(n: int, pd: float = 0.01, rho: float = 0.15, lgd: float = 0.45) -> Portfolio:
    z = np.zeros(n, dtype=np.int64)
    return Portfolio(np.full(n, pd), np.ones(n), np.full(n, lgd), z, z.copy(), np.array([rho]), 0.0)


def test_basel_rho_limits() -> None:
    assert basel_corporate_rho(1e-9) == pytest.approx(0.24, abs=1e-6)
    assert basel_corporate_rho(0.999) == pytest.approx(0.12, abs=1e-6)
    r = basel_corporate_rho(np.linspace(0.0003, 0.2, 50))
    assert np.all(np.diff(r) < 0)


def test_irb_k_structure() -> None:
    """K(M=2.5) = ASRF K / (1 - 1.5 b); K grows with M (CRE31.5/31.6)."""
    pd, lgd = 0.01, 0.45
    r = basel_corporate_rho(pd)
    b = maturity_adjustment_b(pd)
    assert irb_k(pd, lgd, 2.5) == pytest.approx(asrf_k(pd, lgd, r) / (1 - 1.5 * b))
    assert irb_k(pd, lgd, 1.0) < irb_k(pd, lgd, 5.0)


def test_asrf_equals_vasicek_quantile() -> None:
    pd, rho, lgd = 0.02, 0.12, 0.4
    assert asrf_k(pd, lgd, rho) == pytest.approx(
        vasicek_loss_quantile(0.999, pd, rho, lgd) - pd * lgd, rel=1e-12
    )
    x = vasicek_loss_quantile(np.array([0.5, 0.99]), pd, rho, lgd)
    assert vasicek_loss_cdf(x, pd, rho, lgd) == pytest.approx([0.5, 0.99], rel=1e-9)


def test_mc_matches_asrf_cpu() -> None:
    """Granular homogeneous portfolio: simulated 99.9% UL vs analytic ASRF (CPU subset)."""
    n, s = 20000, 60000
    pf = homogeneous(n)
    rm = risk_measures(pf, simulate(pf, s, seed=11, device="cpu", chunk=6000), 0.999)
    k = asrf_k(0.01, 0.45, 0.15)
    assert (rm.var - rm.el_analytic) / n == pytest.approx(k, rel=0.06)


def test_mc_matches_vasicek_body_cpu() -> None:
    n, s = 20000, 60000
    pf = homogeneous(n, pd=0.02, rho=0.1)
    loss = simulate(pf, s, seed=12, device="cpu", chunk=6000).loss / n
    for q in (0.5, 0.9, 0.99):
        assert np.quantile(loss, q) == pytest.approx(
            vasicek_loss_quantile(q, 0.02, 0.1, 0.45), rel=0.03
        )


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
def test_mc_matches_asrf_gpu_tight() -> None:
    n, s = 50000, 1_000_000
    pf = homogeneous(n)
    rm = risk_measures(pf, simulate(pf, s, seed=13, device="cuda"), 0.999)
    assert (rm.var - rm.el_analytic) / n == pytest.approx(asrf_k(0.01, 0.45, 0.15), rel=0.015)


@pytest.fixture(scope="module")
def real_pf() -> Portfolio:
    lb = pl.read_parquet(SYNTH_DIR / "loan_book.parquet")
    return build_portfolio(lb, load_config("model")["credit"])


def test_euler_es_contributions_sum_to_es(real_pf: Portfolio) -> None:
    res = simulate(real_pf, 8000, seed=3, device="cpu", chunk=4000)
    rm = risk_measures(real_pf, res, 0.99)
    assert rm.group_es_contrib.sum() == pytest.approx(rm.es, rel=1e-9)
    assert rm.group_var_contrib.sum() == pytest.approx(rm.var, rel=1e-9)
    assert rm.group_el.sum() == pytest.approx(np.sum(real_pf.pd * real_pf.lgd * real_pf.ead))


def test_t_copula_preserves_marginal_pd() -> None:
    n, s = 5000, 20000
    pf = homogeneous(n, pd=0.03, rho=0.2)
    loss = simulate(pf, s, seed=5, device="cpu", copula="t", nu=6).loss
    se = loss.std() / np.sqrt(s)
    assert abs(loss.mean() - 0.03 * 0.45 * n) < 4 * se


def test_simulation_deterministic_cpu() -> None:
    pf = homogeneous(3000)
    a = simulate(pf, 5000, seed=9, device="cpu", chunk=2000).loss
    b = simulate(pf, 5000, seed=9, device="cpu", chunk=2000).loss
    assert np.array_equal(a, b)
