"""Project paths and config loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
EXTERNAL_DIR = DATA_DIR / "external"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"


@lru_cache(maxsize=8)
def load_config(name: str = "bank") -> dict[str, Any]:
    with (CONFIG_DIR / f"{name}.yaml").open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg
