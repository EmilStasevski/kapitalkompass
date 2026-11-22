"""Deterministic writers for results/ (the single source of every reported number)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kapitalkompass.config import RESULTS_DIR


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, np.floating | float):
        return float(f"{float(o):.10g}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(rel: str, payload: dict[str, Any]) -> Path:
    path = RESULTS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")
    return path


def read_json(rel: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((RESULTS_DIR / rel).read_text())
    return out


def write_table(rel: str, df: pl.DataFrame) -> Path:
    """CSV, rounded to 10 significant digits: diffable and byte-stable."""
    path = RESULTS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(
        [
            pl.col(c).map_elements(lambda x: float(f"{x:.10g}"), return_dtype=pl.Float64)
            for c, t in df.schema.items()
            if t in (pl.Float64, pl.Float32)
        ]
    )
    df.write_csv(path)
    return path


def read_table(rel: str) -> pl.DataFrame:
    return pl.read_csv(RESULTS_DIR / rel)
