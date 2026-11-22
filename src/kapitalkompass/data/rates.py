"""ECB AAA euro-area spot curve (Svensson), fetched once and cached as parquet.

The pipeline reads only the cache, so results do not depend on network state.
`task data:rates` refreshes the cache explicitly.
"""

from __future__ import annotations

import io
from datetime import date

import httpx
import polars as pl

from kapitalkompass.config import EXTERNAL_DIR, load_config
from kapitalkompass.data.schemas import RateCurve

ECB_URL = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{keys}"
CACHE = EXTERNAL_DIR / "ecb_yc_spot.parquet"


def tenor_years(tenor: str) -> float:
    n, unit = int(tenor[:-1]), tenor[-1]
    return n / 12.0 if unit == "M" else float(n)


def fetch(start: str, tenors: list[str], end: str | None = None) -> pl.DataFrame:
    keys = "+".join(f"SR_{t}" for t in tenors)
    params = {"format": "csvdata", "startPeriod": start, "detail": "dataonly"}
    if end:
        params["endPeriod"] = end
    r = httpx.get(ECB_URL.format(keys=keys), params=params, timeout=120.0)
    r.raise_for_status()
    raw = pl.read_csv(io.StringIO(r.text), infer_schema_length=0)
    df = raw.select(
        date=pl.col("TIME_PERIOD").str.to_date("%Y-%m-%d"),
        tenor=pl.col("DATA_TYPE_FM").str.replace("SR_", ""),
        rate=pl.col("OBS_VALUE").cast(pl.Float64) / 100.0,
    ).with_columns(tenor_years=pl.col("tenor").map_elements(tenor_years, return_dtype=pl.Float64))
    df = df.select("date", "tenor", "tenor_years", "rate").sort("date", "tenor_years")
    return RateCurve.validate(df)


def refresh() -> pl.DataFrame:
    cfg = load_config("bank")
    df = fetch(cfg["rates"]["start"], cfg["rates"]["tenors"], end=cfg["reporting_date"])
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CACHE, statistics=False)
    return df


def load_curves() -> pl.DataFrame:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing; run `task data:rates` once (needs network)")
    return RateCurve.validate(pl.read_parquet(CACHE))


def curve_at(asof: date | str | None = None) -> pl.DataFrame:
    """Spot curve on the last business day on or before `asof` (default: reporting date)."""
    if asof is None:
        asof = load_config("bank")["reporting_date"]
    if isinstance(asof, str):
        asof = date.fromisoformat(asof)
    df = load_curves().filter(pl.col("date") <= asof)
    last = df["date"].max()
    return df.filter(pl.col("date") == last).sort("tenor_years")


if __name__ == "__main__":
    out = refresh()
    print(out.height, "rows,", out["date"].min(), "to", out["date"].max())
    print(curve_at())
