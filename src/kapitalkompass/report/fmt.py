"""Number formatting for English (README, reports/*.md) and German (README.de.md, PDF)."""

from __future__ import annotations

from typing import Any


class Fmt:
    def __init__(self, lang: str = "en") -> None:
        self.de = lang == "de"

    def num(self, x: float, d: int = 1) -> str:
        s = f"{x:,.{d}f}"
        if self.de:
            s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
        return s.replace("-", "−")

    def whole(self, x: float) -> str:
        return self.num(round(x), 0)

    def pct(self, x: float, d: int = 1) -> str:
        return self.num(100 * x, d) + (" %" if self.de else "%")

    def pp(self, x: float, d: int = 1) -> str:
        return self.num(100 * x, d) + (" Pp." if self.de else " pp")

    def mio(self, x: float, d: int = 1) -> str:
        return self.num(x / 1e6, d)

    def eur_m(self, x: float, d: int = 1) -> str:
        return self.mio(x, d) + (" Mio. EUR" if self.de else "m EUR")

    def yes(self, b: Any) -> str:
        return ("ja" if b else "nein") if self.de else ("yes" if b else "no")


STATUS_DE = {"green": "grün", "amber": "gelb", "red": "rot"}
LIMIT_NAMES = {
    "credit_risk_capital": ("Credit risk capital", "Kreditrisikokapital"),
    "irrbb_eve": ("IRRBB ΔEVE", "Zinsrisiko ΔEVE"),
    "total_risk_cover": ("Total risk cover", "Risikodeckungsmasse gesamt"),
    "nii_shock": ("NII shock", "Zinsergebnis-Schock"),
    "sector_concentration": ("Sector concentration", "Sektorkonzentration"),
    "single_name": ("Single name", "Einzelkredit"),
    "lcr": ("LCR (internal floor)", "LCR (interne Untergrenze)"),
}
SCENARIO_DE = {
    "parallel_up": "Parallel aufwärts",
    "parallel_down": "Parallel abwärts",
    "steepener": "Versteilerung",
    "flattener": "Verflachung",
    "short_up": "Kurzfristig aufwärts",
    "short_down": "Kurzfristig abwärts",
}
