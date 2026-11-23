"""Static figures for the PDF and the Markdown reports (matplotlib, deterministic PNGs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kapitalkompass.config import REPORTS_DIR
from kapitalkompass.report.facts import Facts
from kapitalkompass.report.fmt import SCENARIO_DE

FIG_DIR = REPORTS_DIR / "figures"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#c9c8c3",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "svg.hashsalt": "kapitalkompass",
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=200, metadata={"Software": None})
    plt.close(fig)
    return p


def _de(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def loss_distribution(f: Facts) -> Path:
    import polars as pl

    from kapitalkompass.results import read_table

    h: pl.DataFrame = read_table("credit/loss_histogram.csv")
    mid = (h["bin_left_eur"] + h["bin_right_eur"]).to_numpy() / 2e6
    width = float((h["bin_right_eur"] - h["bin_left_eur"]).to_numpy()[0] / 1e6)
    dens = h["count"].to_numpy() / h["count"].sum()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(mid, dens, width=width * 0.9, color=BLUE, linewidth=0)
    ax.set_yscale("log")
    for key, label, yy, ha, dx in (
        ("var_eur", "VaR 99,9 %", 0.75, "right", -4),
        ("es_eur", "ES 99,9 %", 0.45, "left", 4),
    ):
        x = f.credit[key] / 1e6
        ax.axvline(x, color=INK, lw=1, ls="--" if key == "es_eur" else "-")
        ax.annotate(
            f"{label}\n{_de(x, 1)}",
            (x, yy),
            xycoords=("data", "axes fraction"),
            xytext=(dx, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            ha=ha,
        )
    ax.set_xlabel("Portfolioverlust (Mio. EUR, Gauß-Copula)")
    ax.set_ylabel("Anteil der Szenarien (log)")
    return _save(fig, "loss_distribution.png")


def raroc_by_segment(f: Facts) -> Path:
    df = f.raroc.sort("raroc")
    hurdle = f.steer["hurdle_rate"] * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(df.height)
    vals = df["raroc"].to_numpy() * 100
    below = df["below_hurdle"].to_numpy()
    ax.barh(y, vals, color=[ORANGE if b else BLUE for b in below], height=0.7)
    ax.axvline(hurdle, color=INK, lw=1)
    ax.annotate(
        f"Hurdle {_de(hurdle)} %",
        (hurdle, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_yticks(y, df["segment"].to_list(), fontsize=8)
    for yi, v in zip(y, vals, strict=True):
        ax.annotate(
            _de(v, 1),
            (v, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=MUTED,
        )
    ax.set_xlabel("RAROC (%) — orange: unter der Hurdle Rate")
    ax.grid(axis="y", visible=False)
    return _save(fig, "raroc_by_segment.png")


def irrbb(f: Facts) -> Path:
    df = f.irrbb
    names = [SCENARIO_DE[s] for s in df["scenario"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, col, title in (
        (axes[0], "delta_eve_pct_tier1", "ΔEVE in % des Kernkapitals"),
        (axes[1], "delta_nii_pct_tier1", "ΔNII in % des Kernkapitals"),
    ):
        v = df[col].to_numpy() * 100
        ax.barh(names, v, color=[ORANGE if x < 0 else BLUE for x in v], height=0.65)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    thr = f.alm["irrbb"]["eve_threshold_pct_tier1"] * 100
    axes[0].axvline(-thr, color=INK, lw=1, ls="--")
    axes[0].annotate(
        f"−{_de(thr)} %",
        (-thr, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=7,
    )
    nthr = f.alm["irrbb"]["nii_threshold_pct_tier1"] * 100
    axes[1].axvline(-nthr, color=INK, lw=1, ls="--")
    axes[1].annotate(
        f"−{_de(nthr)} %",
        (-nthr, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=7,
    )
    return _save(fig, "irrbb_scenarios.png")


def granularity(f: Facts) -> Path | None:
    if not f.experiments:
        return None
    rows = f.experiments["granularity"]["portfolios"]
    x = [r["hhi_name"] * 1e3 for r in rows]
    y = [r["addon"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5)
    ax.axhline(5, color=INK, lw=0.8, ls="--")
    ax.axvline(3, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Namens-HHI (×1000)")
    ax.set_ylabel("Granularitätsaufschlag (%)")
    return _save(fig, "granularity.png")


def drift(f: Facts) -> Path | None:
    if not f.drift:
        return None
    sev = sorted(f.drift["severities"], key=lambda k: f.drift["severities"][k])  # type: ignore[index]
    labels = {"none": "kein", "mild": "mild", "medium": "mittel", "strong": "stark"}
    a = [f.drift["detection_rate"][s] * 100 for s in sev]
    b = [f.drift["exploratory_any_column_rule"][s] * 100 for s in sev]
    x = np.arange(len(sev))
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(x - 0.18, a, width=0.34, color=BLUE, label="Datensatz-Regel (vorregistriert)")
    ax.bar(x + 0.18, b, width=0.34, color=ORANGE, label="Einzelspalten-Regel (explorativ)")
    ax.set_xticks(x, [labels.get(s, s) for s in sev])
    ax.set_xlabel("Schwere des Immobilienschocks")
    ax.set_ylabel("Batches mit Drift (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    return _save(fig, "drift_detection.png")


def all_figures(f: Facts) -> dict[str, Path | None]:
    return {
        "loss": loss_distribution(f),
        "raroc": raroc_by_segment(f),
        "irrbb": irrbb(f),
        "granularity": granularity(f),
        "drift": drift(f),
    }
