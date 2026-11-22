"""pandera data contracts for every table the pipeline reads or writes."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

SECTORS = ["manufacturing", "trade", "real_estate"]
GRADES = [1, 2, 3, 4]
SEGMENTS = [f"{s}_{g}" for s in SECTORS for g in GRADES]


class LoanBook(pa.DataFrameModel):
    obligor_id: int = pa.Field(unique=True, ge=0)
    sector: str = pa.Field(isin=SECTORS)
    grade: int = pa.Field(isin=GRADES)
    segment: str = pa.Field(isin=SEGMENTS)
    pd: float = pa.Field(gt=0.0, lt=1.0)
    ead_eur: float = pa.Field(gt=0.0)
    lgd: float = pa.Field(ge=0.0, le=1.0)
    maturity_years: float = pa.Field(gt=0.0, le=30.0)
    fixed_rate: bool
    client_margin: float = pa.Field(ge=-0.01, le=0.10)
    fee_rate: float = pa.Field(ge=0.0, le=0.05)
    opex_rate: float = pa.Field(ge=0.0, le=0.05)
    is_large: bool

    class Config:
        strict = True
        coerce = False

    @pa.dataframe_check
    @classmethod
    def segment_matches(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col("segment") == pl.col("sector") + pl.lit("_") + pl.col("grade").cast(pl.Utf8)
        )


class BalanceSheet(pa.DataFrameModel):
    item: str = pa.Field(unique=True)
    side: str = pa.Field(isin=["asset", "liability", "equity"])
    amount_eur: int = pa.Field(ge=0)
    maturity_years: float = pa.Field(ge=0.0, nullable=True)

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def identity_holds(cls, data: pa.PolarsData) -> pl.LazyFrame:
        signed = (
            pl.when(pl.col("side") == "asset")
            .then(pl.col("amount_eur"))
            .otherwise(-pl.col("amount_eur"))
        )
        return data.lazyframe.select(signed.sum() == 0)


class RateCurve(pa.DataFrameModel):
    date: pl.Date
    tenor: str
    tenor_years: float = pa.Field(gt=0.0, le=30.0)
    rate: float = pa.Field(ge=-0.05, le=0.20)  # decimal, continuous compounding

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def unique_key(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(pl.struct("date", "tenor").is_unique().all())
