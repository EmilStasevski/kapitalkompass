"""Module 3 pipeline step: behavioural deposits, ladders, LCR/NSFR, EVE/NII, FTP."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from kapitalkompass.alm.curves import SCENARIOS, ZeroCurve
from kapitalkompass.alm.deposits import (
    Profile,
    behavioural_profile,
    fit_model,
    kaplan_meier,
    simulate_accounts,
)
from kapitalkompass.alm.ftp import ftp_loans
from kapitalkompass.alm.irrbb import Position, amortizing, bullet, floating, run_scenarios
from kapitalkompass.alm.liquidity import ladder, lcr, nsfr
from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.rates import curve_at
from kapitalkompass.data.schemas import BalanceSheet, LoanBook
from kapitalkompass.results import write_json, write_table

DEPOSITS = [
    "retail_deposits_stable",
    "retail_deposits_less_stable",
    "corp_deposits_operational",
    "corp_deposits_non_operational",
]


def base_curve() -> tuple[ZeroCurve, str]:
    c = curve_at()
    return ZeroCurve(c["tenor_years"].to_numpy(), c["rate"].to_numpy()), str(c["date"][0])


def loan_positions(lb: pl.DataFrame, curve: ZeroCurve, reset: float) -> list[Position]:
    pos: list[Position] = []
    fixed = lb.filter(pl.col("fixed_rate")).with_columns(
        m=(pl.col("maturity_years") * 12).ceil().cast(pl.Int64)
    )
    for m, amt in fixed.group_by("m").agg(pl.col("ead_eur").sum()).sort("m").iter_rows():
        mat = m / 12.0
        pos.append(amortizing(f"loans_fixed_{m}m", "asset", amt, mat, float(curve.rate(mat / 2))))
    fl = float(lb.filter(~pl.col("fixed_rate"))["ead_eur"].sum())
    pos.append(floating("loans_floating", "asset", fl, reset, float(curve.rate(reset))))
    return pos


def balance_positions(
    bs: dict[str, float], curve: ZeroCurve, profiles: dict[str, Profile], betas: dict[str, float]
) -> list[Position]:
    on = 1.0 / 365.0
    pos = [
        floating("hqla_l1_cash_cb", "asset", bs["hqla_l1_cash_cb"], on * 2, float(curve.rate(on)))
    ]
    for item, m in [
        ("hqla_l1_gov_bonds", 5.0),
        ("hqla_l2a_covered", 4.0),
        ("hqla_l2b_corp", 3.0),
        ("bonds_non_hqla", 4.0),
    ]:
        pos.append(bullet(item, "asset", bs[item], m, float(curve.rate(m))))
    for item in DEPOSITS:
        t, w = profiles[item].runoff()
        pos.append(
            Position(item, "liability", t, bs[item] * w, [(0.0, bs[item], 0.25, betas[item])])
        )
    n = bs["wholesale_unsecured_lt1y"] / 12.0
    for k in range(12):
        m = (k + 0.5) / 12.0
        pos.append(bullet(f"wholesale_lt1y_{k}", "liability", n, m, float(curve.rate(m))))
    pos.append(
        bullet(
            "covered_bonds_issued",
            "liability",
            bs["covered_bonds_issued"],
            6.0,
            float(curve.rate(6.0)),
        )
    )
    pos.append(
        bullet(
            "wholesale_unsecured_gt1y",
            "liability",
            bs["wholesale_unsecured_gt1y"],
            3.0,
            float(curve.rate(3.0)),
        )
    )
    return pos


def principal_flows(
    lb: pl.DataFrame, bs: dict[str, float]
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Contractual principal flows of assets and non-deposit liabilities (ladder)."""
    flows: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    m = (lb["maturity_years"].to_numpy() * 12).astype(float)
    n = np.maximum(1, np.ceil(m - 1e-9)).astype(int)
    ead = lb["ead_eur"].to_numpy()
    nmax = int(n.max())
    # Linear monthly amortisation: amount per month = ead / n for months 1..n
    per = np.zeros(nmax + 1)
    np.add.at(per, n, ead / n)  # contribution ends at month n
    monthly = np.cumsum(per[::-1])[::-1][1:]  # sum over loans with n >= month
    flows.append(("loans", "asset", np.arange(1, nmax + 1) / 12.0, monthly))
    flows.append(
        ("hqla_l1_cash_cb", "asset", np.array([1 / 365]), np.array([bs["hqla_l1_cash_cb"]]))
    )
    for item, mat in [
        ("hqla_l1_gov_bonds", 5.0),
        ("hqla_l2a_covered", 4.0),
        ("hqla_l2b_corp", 3.0),
        ("bonds_non_hqla", 4.0),
    ]:
        flows.append((item, "asset", np.array([mat]), np.array([bs[item]])))
    flows.append(
        (
            "wholesale_unsecured_lt1y",
            "liability",
            (np.arange(12) + 0.5) / 12,
            np.full(12, bs["wholesale_unsecured_lt1y"] / 12),
        )
    )
    flows.append(
        (
            "covered_bonds_issued",
            "liability",
            np.array([6.0]),
            np.array([bs["covered_bonds_issued"]]),
        )
    )
    flows.append(
        (
            "wholesale_unsecured_gt1y",
            "liability",
            np.array([3.0]),
            np.array([bs["wholesale_unsecured_gt1y"]]),
        )
    )
    return flows


def run() -> dict[str, Any]:
    acfg = load_config("alm")
    seed = int(load_config("model")["seed"])
    lb = LoanBook.validate(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"))
    bsdf = BalanceSheet.validate(pl.read_parquet(SYNTH_DIR / "balance_sheet.parquet"))
    bs = {r["item"]: float(r["amount_eur"]) for r in bsdf.iter_rows(named=True)}
    curve, curve_date = base_curve()
    dcfg, icfg = acfg["deposits"], acfg["irrbb"]

    # --- behavioural deposits
    acc = simulate_accounts(dcfg, seed)
    cats = list(dcfg["categories"])
    model = fit_model(acc, cats)
    profiles: dict[str, Profile] = {}
    surv_rows = []
    check_t = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    for c in cats:
        lbal = acc.filter(pl.col("category") == c)["log_balance"].to_numpy()
        cap = dcfg["bcbs_caps"][dcfg["categories"][c]["bcbs_class"]]
        profiles[c] = behavioural_profile(
            model, c, lbal, cap, dcfg["core_horizon_years"], dcfg["max_years"]
        )
        km = kaplan_meier(acc, c, check_t)
        # account-count survival from the model (equal weights) for a like-for-like KM check
        sf = (
            model.aft.predict_survival_function(_design(lbal, c, cats), times=check_t)
            .to_numpy()
            .mean(1)
        )
        for t, k, s in zip(check_t, km, sf, strict=True):
            surv_rows.append({"category": c, "t_years": t, "kaplan_meier": k, "weibull_aft": s})
    surv = pl.DataFrame(surv_rows)
    write_table("alm/deposit_survival_check.csv", surv)
    prof_tab = pl.DataFrame(
        [
            {
                "category": p.category,
                "balance_eur": bs[p.category],
                "stable_share_model": p.stable_share,
                "core_share": p.core_share,
                "core_cap": dcfg["bcbs_caps"][dcfg["categories"][p.category]["bcbs_class"]][
                    "core_share"
                ],
                "avg_maturity_core_uncapped": p.avg_maturity_uncapped,
                "avg_maturity_core": p.avg_maturity_core,
                "avg_maturity_cap": dcfg["bcbs_caps"][dcfg["categories"][p.category]["bcbs_class"]][
                    "avg_maturity"
                ],
            }
            for p in profiles.values()
        ]
    )
    write_table("alm/deposit_profiles.csv", prof_tab)

    # --- ladders: contractual (NMD overnight) vs behavioural
    pflows = principal_flows(lb, bs)
    contractual = list(pflows) + [
        (d, "liability", np.array([1 / 365]), np.array([bs[d]])) for d in DEPOSITS
    ]
    behavioural = list(pflows)
    for d in DEPOSITS:
        t, w = profiles[d].runoff()
        behavioural.append((d, "liability", t, bs[d] * w))
    lad_c = ladder(contractual, acfg["ladder_buckets_years"])
    lad_b = ladder(behavioural, acfg["ladder_buckets_years"])
    write_table("alm/ladder_contractual.csv", lad_c)
    write_table("alm/ladder_behavioural.csv", lad_b)

    # --- LCR / NSFR
    mat = lb["maturity_years"].to_numpy()
    ead = lb["ead_eur"].to_numpy()
    inflow_30d = float(np.sum(np.where(mat <= 1 / 12, ead, ead / np.ceil(mat * 12))))
    lcr_res = lcr(bs, inflow_30d, acfg["lcr"])
    nsfr_res = nsfr(bs, float(ead[mat < 1].sum()), float(ead[mat >= 1].sum()), acfg["nsfr"])

    # --- IRRBB
    positions = loan_positions(lb, curve, icfg["floating_reprice_years"]) + balance_positions(
        bs, curve, profiles, icfg["deposit_beta"]
    )
    sc = run_scenarios(
        positions,
        curve,
        icfg["shocks_bp"],
        icfg["decay_x"],
        icfg["floor"],
        icfg["nii_horizon_years"],
    )
    tier1 = bs["cet1_equity"]
    irrbb_tab = pl.DataFrame(
        [
            {
                "scenario": k,
                "delta_eve_eur": v["delta_eve"],
                "delta_eve_pct_tier1": v["delta_eve"] / tier1,
                "delta_nii_eur": v["delta_nii"],
                "delta_nii_pct_tier1": v["delta_nii"] / tier1,
            }
            for k, v in sc.items()
            if k in SCENARIOS
        ]
    )
    write_table("alm/irrbb_scenarios.csv", irrbb_tab)
    worst_eve = min(SCENARIOS, key=lambda k: sc[k]["delta_eve"])
    worst_nii = min(SCENARIOS, key=lambda k: sc[k]["delta_nii"])

    # --- FTP
    ftp = ftp_loans(
        lb,
        curve,
        acfg["ftp"]["lp_long"],
        acfg["ftp"]["lp_tau_years"],
        icfg["floating_reprice_years"],
    )
    seg = (
        ftp.group_by("segment")
        .agg(
            pl.col("ead_eur").sum(),
            pl.col("interest_income_eur").sum(),
            pl.col("ftp_charge_eur").sum(),
            pl.col("nim_after_ftp_eur").sum(),
            ((pl.col("ftp_rate") * pl.col("ead_eur")).sum() / pl.col("ead_eur").sum()).alias(
                "ftp_rate_avg"
            ),
            (
                (pl.col("liquidity_premium") * pl.col("ead_eur")).sum() / pl.col("ead_eur").sum()
            ).alias("lp_avg"),
        )
        .sort("segment")
    )
    write_table("alm/ftp_segments.csv", seg)
    lp_curve = (
        pl.DataFrame({"tenor_years": curve.t})
        .with_columns(
            base_rate=curve.r,
            liquidity_premium=acfg["ftp"]["lp_long"]
            * (1 - np.exp(-curve.t / acfg["ftp"]["lp_tau_years"])),
        )
        .with_columns(ftp_rate=pl.col("base_rate") + pl.col("liquidity_premium"))
    )
    write_table("alm/ftp_curve.csv", lp_curve)

    def cum_at(lad: pl.DataFrame, bucket: str) -> float:
        return float(lad.filter(pl.col("bucket") == bucket)["cumulative_gap_eur"].to_numpy()[0])

    summary = {
        "curve_date": curve_date,
        "tier1_eur": tier1,
        "lcr": {k: v for k, v in lcr_res.items()},
        "nsfr": {k: v for k, v in nsfr_res.items()},
        "irrbb": {
            "scenarios": sc,
            "worst_eve_scenario": worst_eve,
            "worst_delta_eve_eur": sc[worst_eve]["delta_eve"],
            "worst_delta_eve_pct_tier1": sc[worst_eve]["delta_eve"] / tier1,
            "eve_outlier": -sc[worst_eve]["delta_eve"] > icfg["eve_outlier_tier1"] * tier1,
            "eve_threshold_pct_tier1": icfg["eve_outlier_tier1"],
            "worst_nii_scenario": worst_nii,
            "worst_delta_nii_eur": sc[worst_nii]["delta_nii"],
            "worst_delta_nii_pct_tier1": sc[worst_nii]["delta_nii"] / tier1,
            "nii_flag": -sc[worst_nii]["delta_nii"] > icfg["nii_outlier_tier1"] * tier1,
            "nii_threshold_pct_tier1": icfg["nii_outlier_tier1"],
            "nii_threshold_verified": False,
        },
        "deposits": {
            "accounts": acc.height,
            "closed_share": float(acc["closed"].to_numpy().mean()),
            "aft_coef_log_balance": float(model.aft.params_.loc[("lambda_", "log_balance")]),
            "max_abs_km_vs_model": float(
                np.abs(surv["kaplan_meier"].to_numpy() - surv["weibull_aft"].to_numpy()).max()
            ),
            "core_share_weighted": float(
                sum(profiles[d].core_share * bs[d] for d in DEPOSITS) / sum(bs[d] for d in DEPOSITS)
            ),
        },
        "ladder": {
            "cum_gap_1y_contractual_eur": cum_at(lad_c, "0.5-1y"),
            "cum_gap_1y_behavioural_eur": cum_at(lad_b, "0.5-1y"),
            "cum_gap_1m_contractual_eur": cum_at(lad_c, "0-0.0833y"),
            "cum_gap_1m_behavioural_eur": cum_at(lad_b, "0-0.0833y"),
        },
        "ftp": {
            "total_ftp_charge_eur": float(seg["ftp_charge_eur"].to_numpy().sum()),
            "total_nim_after_ftp_eur": float(seg["nim_after_ftp_eur"].to_numpy().sum()),
            "avg_ftp_rate": float(
                seg["ftp_charge_eur"].to_numpy().sum() / seg["ead_eur"].to_numpy().sum()
            ),
        },
    }
    write_json("alm/summary.json", summary)
    print(
        f"alm: LCR {lcr_res['lcr']:.0%}, NSFR {nsfr_res['nsfr']:.0%}; worst dEVE {worst_eve} "
        f"{sc[worst_eve]['delta_eve'] / 1e6:.1f}m ({sc[worst_eve]['delta_eve'] / tier1:.1%} T1); "
        f"worst dNII {worst_nii} {sc[worst_nii]['delta_nii'] / 1e6:.1f}m"
    )
    return summary


def _design(lbal: np.ndarray, category: str, cats: list[str]) -> Any:
    import pandas as pd

    df = pd.DataFrame({"log_balance": lbal})
    for c in cats[1:]:
        df[f"cat_{c}"] = 1.0 if c == category else 0.0
    return df


if __name__ == "__main__":
    run()
