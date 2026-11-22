"""Seeded synthetic bank: loan book (Polars) and balance sheet.

Every assumption lives in config/bank.yaml and is documented in docs/data.md.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import polars as pl

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.data.schemas import BalanceSheet, LoanBook


def _beta_params(mean: np.ndarray, concentration: float) -> tuple[np.ndarray, np.ndarray]:
    return mean * concentration, (1.0 - mean) * concentration


def generate_loan_book(cfg: dict[str, Any], seed: int) -> pl.DataFrame:
    lb = cfg["loan_book"]
    rng = np.random.default_rng(seed)
    n = int(lb["n_obligors"])

    sectors = list(lb["sectors"])
    sector_share = np.array([lb["sectors"][s]["share"] for s in sectors])
    grades = [int(g) for g in lb["grades"]]
    grade_share = np.array([lb["grades"][str(g)]["share"] for g in grades])

    sector_idx = rng.choice(len(sectors), size=n, p=sector_share)
    grade_idx = rng.choice(len(grades), size=n, p=grade_share)
    sector = np.array(sectors)[sector_idx]
    grade = np.array(grades)[grade_idx]

    pd_ = np.array([lb["grades"][str(g)]["pd"] for g in grades])[grade_idx]
    margin_base = np.array([lb["grades"][str(g)]["margin"] for g in grades])[grade_idx]
    lgd_mean = np.array([lb["sectors"][s]["lgd_mean"] for s in sectors])[sector_idx]
    mat_mean = np.array([lb["sectors"][s]["maturity_mean"] for s in sectors])[sector_idx]
    opex = np.array([lb["sectors"][s]["opex_rate"] for s in sectors])[sector_idx]

    ex = lb["exposure"]
    ead = rng.lognormal(np.log(ex["lognormal_median_eur"]), ex["lognormal_sigma"], size=n)

    # Large names: replace the first n_large obligors in the allowed sectors with Pareto draws.
    is_large = np.zeros(n, dtype=bool)
    allowed = np.flatnonzero(np.isin(sector, ex["large_sectors"]))
    large_idx = rng.choice(allowed, size=int(ex["n_large"]), replace=False)
    pareto = ex["large_pareto_min_eur"] * (
        1.0 + rng.pareto(ex["large_pareto_alpha"], len(large_idx))
    )
    ead[large_idx] = np.minimum(pareto, ex["large_cap_eur"])
    is_large[large_idx] = True

    a, b = _beta_params(lgd_mean, float(lb["lgd_concentration"]))
    lgd = np.clip(rng.beta(a, b), *lb["lgd_bounds"])

    # Gamma with shape 2 around the sector mean maturity.
    maturity = np.clip(rng.gamma(2.0, mat_mean / 2.0), *lb["maturity_bounds"])
    fixed = rng.random(n) < lb["fixed_rate_share"]
    margin = margin_base + rng.normal(0.0, lb["margin_noise_sd"], n)
    fee = rng.exponential(lb["fee_rate_mean"], n).clip(0.0, 0.05)

    df = pl.DataFrame(
        {
            "obligor_id": np.arange(n, dtype=np.int64),
            "sector": sector,
            "grade": grade.astype(np.int64),
            "pd": pd_,
            "ead_eur": np.round(ead, 2),
            "lgd": np.round(lgd, 6),
            "maturity_years": np.round(maturity, 4),
            "fixed_rate": fixed,
            "client_margin": np.round(margin, 6),
            "fee_rate": np.round(fee, 6),
            "opex_rate": opex,
            "is_large": is_large,
        }
    ).with_columns(
        segment=pl.col("sector") + pl.lit("_") + pl.col("grade").cast(pl.Utf8),
    )
    df = df.select(list(LoanBook.to_schema().columns))
    return LoanBook.validate(df)


def generate_balance_sheet(cfg: dict[str, Any], loan_book: pl.DataFrame) -> pl.DataFrame:
    bs = cfg["balance_sheet"]
    loans = round(float(loan_book["ead_eur"].sum()))
    total_assets = round(loans / bs["loans_share"])
    w_loans = float((loan_book["ead_eur"] * loan_book["maturity_years"]).sum()) / loans

    rows: list[dict[str, Any]] = [
        {"item": "loans", "side": "asset", "amount_eur": loans, "maturity_years": round(w_loans, 4)}
    ]
    for item, spec in bs["assets"].items():
        rows.append(
            {
                "item": item,
                "side": "asset",
                "amount_eur": round(total_assets * spec["share"]),
                "maturity_years": spec["maturity"],
            }
        )
    asset_sum = sum(r["amount_eur"] for r in rows)
    eq_item, eq_spec = next(iter(bs["equity"].items()))
    equity = round(total_assets * eq_spec["share"])
    liab_rows: list[dict[str, Any]] = []
    residual_item = None
    for item, spec in bs["liabilities"].items():
        if spec["share"] is None:
            residual_item = (item, spec)
            continue
        liab_rows.append(
            {
                "item": item,
                "side": "liability",
                "amount_eur": round(total_assets * spec["share"]),
                "maturity_years": spec["maturity"],
            }
        )
    assert residual_item is not None, "one liability must be the residual"
    residual = asset_sum - equity - sum(r["amount_eur"] for r in liab_rows)
    if residual < 0:
        raise ValueError("balance sheet shares exceed total assets")
    liab_rows.append(
        {
            "item": residual_item[0],
            "side": "liability",
            "amount_eur": residual,
            "maturity_years": residual_item[1]["maturity"],
        }
    )
    eq_row = {"item": eq_item, "side": "equity", "amount_eur": equity, "maturity_years": None}
    df = pl.DataFrame(
        rows + liab_rows + [eq_row],
        schema={
            "item": pl.Utf8,
            "side": pl.Utf8,
            "amount_eur": pl.Int64,
            "maturity_years": pl.Float64,
        },
    )
    return BalanceSheet.validate(df)


def frame_hash(df: pl.DataFrame) -> str:
    """Content hash independent of parquet metadata."""
    return hashlib.sha256(df.write_csv().encode()).hexdigest()


def build(seed: int | None = None, write: bool = True) -> dict[str, pl.DataFrame]:
    cfg = load_config("bank")
    seed = int(cfg["seed"]) if seed is None else seed
    loans = generate_loan_book(cfg, seed)
    bs = generate_balance_sheet(cfg, loans)
    from kapitalkompass.data.history import generate_history

    out = {
        "loan_book": loans,
        "balance_sheet": bs,
        "default_history": generate_history(cfg, seed),
    }
    if write:
        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        for name, df in out.items():
            df.write_parquet(SYNTH_DIR / f"{name}.parquet", statistics=False)
            lines.append(f"{frame_hash(df)}  {name}")
        (SYNTH_DIR / "HASHES").write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    out = build()
    for name, df in out.items():
        print(f"{name}: {df.height} rows, sha256 {frame_hash(df)[:12]}")


if __name__ == "__main__":
    main()
