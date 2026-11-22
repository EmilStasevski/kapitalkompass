"""RAROC per segment = (NIM after FTP + fees - opex - EL) / economic capital (Euler)."""

from __future__ import annotations

import polars as pl


def raroc_table(
    lb: pl.DataFrame, ftp_seg: pl.DataFrame, capital: pl.DataFrame, hurdle: float
) -> pl.DataFrame:
    base = lb.group_by("segment", "sector").agg(
        fees_eur=(pl.col("fee_rate") * pl.col("ead_eur")).sum(),
        opex_eur=(pl.col("opex_rate") * pl.col("ead_eur")).sum(),
        el_eur=(pl.col("pd") * pl.col("lgd") * pl.col("ead_eur")).sum(),
        largest_name_eur=pl.col("ead_eur").max(),
    )
    t = (
        base.join(
            ftp_seg.select("segment", "ead_eur", "nim_after_ftp_eur", "ftp_charge_eur"),
            on="segment",
        )
        .join(capital.select("segment", "economic_capital_eur"), on="segment")
        .with_columns(
            risk_adjusted_profit_eur=pl.col("nim_after_ftp_eur")
            + pl.col("fees_eur")
            - pl.col("opex_eur")
            - pl.col("el_eur")
        )
        .with_columns(
            raroc=pl.col("risk_adjusted_profit_eur") / pl.col("economic_capital_eur"),
            eva_eur=pl.col("risk_adjusted_profit_eur") - hurdle * pl.col("economic_capital_eur"),
        )
        .with_columns(below_hurdle=pl.col("raroc") < hurdle)
        .sort("segment")
    )
    return t


def bank_raroc(t: pl.DataFrame) -> float:
    return float(
        t["risk_adjusted_profit_eur"].to_numpy().sum() / t["economic_capital_eur"].to_numpy().sum()
    )
