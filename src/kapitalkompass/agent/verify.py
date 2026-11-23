"""Deterministic verification of agent output (no LLM involved).

Checks, for a Vorstandsvorlage:
- every number in the text matches a numeric value in the tool results of this session,
  up to the rounding the text itself shows (``15,9`` matches 15.94; ``16`` matches 15.94;
  ``15,90`` does not match 15.94); signs are ignored (a "Rückgang um 152,5 Mio." is fine);
- every citation id (``[crr.art395.1]`` in the text and in ``zitate``) was retrieved by
  ``search_rules`` in this session, and at least one citation is present;
- dates equal a date returned by a tool;
- the rendered Vorlage fits on one page.
Regulatory reference numbers (Art. 395, BTR 3.1, LCR30.40, 2024/856, Tier 1 ...) and segment
identifiers (trade_4) are not figures and are removed before number extraction.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from kapitalkompass.config import load_config

CITATION = re.compile(r"\[([a-z0-9]+(?:\.[a-z0-9]+)+(?:#\d+)?(?:~\d+)?)\]")
REFERENCES = [
    r"\b(?:Art\.|Artikel|Article|Abs\.|Absatz|Tz\.|Textziffer|Rn\.|Randnummer|para\.?|Nr\.|§|Ziffer|Annex|Anhang|Unterabs\.)\s*\d+[a-z]*(?:\s*\(\d+[a-z]?\))*(?:\s*(?:Buchst\.|lit\.)\s*[a-z]\))?",
    r"\((?:\d+[a-z]?|[a-z])\)",  # "(1)", "(b)" after an article reference
    r"\b(?:AT|BTO|BTR|BT)\s*\d+(?:\.\d+)*",
    r"\b(?:LCR|NSF|CRE|SRP)\d{2}(?:\.\d+)?",
    r"\bEBA/GL/\d{4}/\d+",
    r"\bBCBS\s*\d+",
    r"\b\d{1,4}/\d{2,4}(?:/E[UGW]+)?\b",  # 575/2013, 2013/36/EU, 2024/856, 05/2023
    r"\b(?:Tier|Level|Säule|Pillar|Stufe)\s*\d[AB]?\b",
    r"\b(?:CET|AT)1\b",
    r"\bbge-m3\b",
    r"\b[a-z]+(?:_[a-z]+)*_\d\b",  # segment ids such as real_estate_2
]
DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b|\b(\d{4})-(\d{2})-(\d{2})\b")
NUMBER = re.compile(
    r"(?<![\w])[-−+]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?:\s*(Mrd|Mio|Tsd|bp|Bp|Basispunkt))?"
)


# Spelled-out numbers are figures too ("zwei Limite sind rot"). "ein/eine" is left out
# because it is also the indefinite article.
WORDS = {
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
}
WORD = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)


@dataclass
class Report:
    numbers: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    bad_dates: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    has_citation: bool = True
    one_page: bool = True
    passed: bool = True

    def feedback(self) -> str:
        msg = []
        if self.unverified_numbers:
            msg.append(
                "Diese Zahlen stehen in keinem Tool-Ergebnis: "
                + ", ".join(self.unverified_numbers)
                + ". Verwende nur Zahlen wörtlich aus den Tool-Ergebnissen oder lass sie weg."
            )
        if self.invalid_citations:
            msg.append("Ungültige Zitate (nicht abgerufen): " + ", ".join(self.invalid_citations))
        if not self.has_citation:
            msg.append("Mindestens ein Zitat [chunk-id] aus search_rules ist erforderlich.")
        if self.contradictions:
            msg.append("Widerspruch zu Tool-Ergebnissen: " + " | ".join(self.contradictions))
        if self.bad_dates:
            msg.append("Nicht belegte Daten: " + ", ".join(self.bad_dates))
        if not self.one_page:
            msg.append("Die Vorlage ist länger als eine Seite; kürzen.")
        return " ".join(msg)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def numeric_leaves(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(numeric_leaves(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(numeric_leaves(v))
    return out


def string_leaves(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in string_leaves(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in string_leaves(v)]
    return []


def _iso(m: re.Match[str]) -> str:
    if m.group(1):
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return f"{m.group(4)}-{m.group(5)}-{m.group(6)}"


def parse_number(tok: str) -> list[tuple[float, int]]:
    """Candidate (value, decimals shown) readings of a German or English formatted number."""
    s = tok.replace("−", "-").replace("+", "")
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        # German thousands separators; '1.234' could also be English 1.234 -> both readings
        dec = len(s.split(",")[1]) if "," in s else 0
        readings = [(float(s.replace(".", "").replace(",", ".")), dec)]
        if s.count(".") == 1 and "," not in s:
            readings.append((float(s), 3))
        return readings
    if "," in s:
        return [(float(s.replace(",", ".")), len(s.split(",")[1]))]
    if "." in s:
        return [(float(s), len(s.split(".")[1]))]
    return [(float(s), 0)]


def matches(value: float, decimals: int, candidates: list[float], rel_tol: float) -> bool:
    r = abs(value)
    half = 0.5 * 10.0 ** (-decimals) + 1e-9
    for v in candidates:
        for c in (abs(v), abs(v) * 100.0, abs(v) / 100.0):
            if abs(r - c) <= max(half, rel_tol * c):
                return True
    return False


def strip_references(text: str) -> str:
    t = CITATION.sub(" ", text)
    for p in REFERENCES:
        t = re.sub(p, " ", t)
    return t


def _num(m: re.Match[str]) -> str:
    """Numeric part of a NUMBER match (without the unit word)."""
    tok = m.group(0).strip()
    return tok[: len(tok) - len(m.group(1))].strip() if m.group(1) else tok


def check_numbers(
    text: str, candidates: list[float], allowed_dates: set[str], rel_tol: float
) -> tuple[list[str], list[str], list[str]]:
    t = strip_references(text)
    bad_dates = []
    for m in DATE.finditer(t):
        if _iso(m) not in allowed_dates:
            bad_dates.append(m.group(0))
    t = DATE.sub(" ", t)
    found, bad = [], []
    for m in NUMBER.finditer(t):
        tok, unit = m.group(0).strip(), (m.group(1) or "")
        num = tok[: len(tok) - len(unit)].strip() if unit else tok
        found.append(tok)
        ok = False
        for v, d in parse_number(num):
            scaled = [(v, d)]
            if unit == "Mrd":
                scaled = [(v * 1000.0, d - 3)]  # Mrd -> Mio, rounding precision scales too
            elif unit == "Tsd":
                scaled = [(v / 1000.0, d + 3)]
            elif abs(v) >= 1e5 and not unit:
                scaled.append((v / 1e6, d + 6))  # full EUR amounts vs Mio. EUR tool values
            if unit in ("bp", "Bp", "Basispunkt"):
                scaled.append((v / 100.0, d + 2))  # basis points vs percent
            if any(matches(sv, max(sd, -6), candidates, rel_tol) for sv, sd in scaled):
                ok = True
                break
        if not ok:
            bad.append(tok)
    for m in WORD.finditer(t):
        found.append(m.group(0))
        if not matches(float(WORDS[m.group(1).lower()]), 0, candidates, rel_tol):
            bad.append(m.group(0))
    return found, bad, bad_dates


# Polarity check for the supervisory outlier claims (EVE outlier, NII large decline). A sentence
# about the test is checked against the boolean flags of the scenario it names (from run_shock),
# or against the bank-level flag if it names no scenario. Heuristic and deliberately narrow.
CLAIM_TESTS = {
    "eve": (
        r"EVE|Ausreißer|outlier",
        "eve_outlier_in_this_scenario",
        "bank_is_eve_outlier",
    ),
    "nii": (
        r"NII|Zinsergebnis|Zinsüberschuss|large decline",
        "nii_large_decline_in_this_scenario",
        "bank_has_nii_large_decline",
    ),
}
NEGATED = re.compile(
    r"nicht\s+(?:\w+\s+){0,2}(?:überschr|übersteig|auffällig|überzogen|erreich)|unterschreit|"
    r"\bweder\b.*\bnoch\b|"
    r"unterhalb|kein\w*\s+(?:\w+\s+)?(?:Ausreißer|Überschreitung|large decline)|eingehalten|"
    r"innerhalb\s+(?:der|des)\s+(?:\w+\s+)?(?:\w*schwelle|\w*grenze|Schwellenwert|Limit)|"
    r"(?:fällt|liegt|bleibt|befindet\s+sich)\s+(?:\S+\s+){0,5}unter\s+(?:der|die|dem|den)\s+"
    r"(?:\w+\s+){0,2}(?:\w*schwelle|\w*grenze|Schwellenwert)|"
    r"unter\s+(?:der|die|dem)\s+(?:\w*schwelle|Schwellenwert)\w*\s+(?:\S+\s+){0,7}(?:fällt|liegt|bleibt)",
    re.IGNORECASE,
)
AFFIRMED = re.compile(
    r"(?<!nicht )(?:überschreit|übersteig|überschritten|ist\s+(?:ein\s+)?Ausreißer|auffällig)",
    re.IGNORECASE,
)


ABBREV = re.compile(r"\b(Mio|Mrd|Tsd|Tz|Art|Rn|Abs|Nr|ggf|bzw|inkl|ca|vgl|Buchst|z\. B|d\. h)\.")


def _sentences(text: str) -> list[str]:
    protected = ABBREV.sub(lambda m: m.group(1) + "§", text)
    return [s.replace("§", ".") for s in re.split(r"(?<=[.!?])\s+|\n", protected)]


BANK_WORDS = re.compile(r"\b(?:Bank|Institut|Gesamtbank|insgesamt)\b", re.IGNORECASE)


def check_claims(text: str, tool_results: list[dict[str, Any]]) -> list[str]:
    shocks = [r for r in tool_results if "eve_outlier_in_this_scenario" in r]
    if not shocks:
        return []
    by_scenario = {r["scenario"]: r for r in shocks}
    out = []
    for sent in _sentences(text):
        if not re.search(
            r"Schwelle|Ausreißer|outlier|large decline|überschr|übersteig", sent, re.IGNORECASE
        ):
            continue
        named = [s for s in by_scenario if s in sent]
        bank_level = bool(BANK_WORDS.search(sent)) and not named
        for trig, scen_flag, bank_flag in CLAIM_TESTS.values():
            if not re.search(trig, sent, re.IGNORECASE):
                continue
            if bank_level:
                flags = [bool(shocks[0][bank_flag])]
            else:  # named scenarios, else all scenarios queried in this session
                flags = [bool(by_scenario[s][scen_flag]) for s in (named or by_scenario)]
            # a denial is wrong only if every scenario in question breaches; an affirmation
            # only if none does
            truth_all, truth_any = all(flags), any(flags)
            neg, pos = bool(NEGATED.search(sent)), bool(AFFIRMED.search(sent))
            if truth_all and neg and not pos:
                out.append(f"'{sent.strip()}' verneint eine Schwellenüberschreitung, die vorliegt")
            elif not truth_any and pos and not neg:
                out.append(f"'{sent.strip()}' behauptet eine Überschreitung, die nicht vorliegt")
    return out


def verify_answer(
    answer: dict[str, Any],
    tool_results: list[dict[str, Any]],
    retrieved: dict[str, str],
    rendered: str,
) -> Report:
    """``retrieved`` maps chunk id -> chunk text returned by search_rules in this session.
    Figures quoted from a rule (e.g. "25 % des Kernkapitals") verify only against the text of
    chunks the answer actually cites."""
    cfg = load_config("agent")
    rel_tol = float(cfg["verify"]["rel_tol"])
    max_chars = int(cfg["verify"].get("max_page_chars", 3800))
    text = "\n".join(
        [
            answer.get("titel", ""),
            answer.get("lage", ""),
            *list(answer.get("treiber", [])),
            *list(answer.get("empfehlung", [])),
        ]
    )
    cited = list(dict.fromkeys(CITATION.findall(text) + list(answer.get("zitate", []))))
    candidates = [v for r in tool_results for v in numeric_leaves(r)]
    # figures inside tool strings (e.g. "within 10 business days"); rule texts are handled below
    for r in tool_results:
        if "chunks" not in r:
            for st in string_leaves(r):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", st):
                    continue  # dates are checked separately
                candidates += [
                    v
                    for m in NUMBER.finditer(strip_references(st))
                    for v, _ in parse_number(_num(m))
                ]
    for cid in cited:
        if cid in retrieved:
            body = strip_references(retrieved[cid])
            candidates += [v for m in NUMBER.finditer(body) for v, _ in parse_number(_num(m))]
    dates = {
        s for r in tool_results for s in string_leaves(r) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s)
    }
    rep = Report()
    rep.numbers, rep.unverified_numbers, rep.bad_dates = check_numbers(
        text, candidates, dates, rel_tol
    )
    rep.contradictions = check_claims(text, tool_results)
    rep.citations = cited
    rep.invalid_citations = [c for c in cited if c not in retrieved]
    rep.has_citation = bool(cited)
    rep.one_page = len(rendered) <= max_chars
    rep.passed = (
        not rep.unverified_numbers
        and not rep.invalid_citations
        and rep.has_citation
        and not rep.bad_dates
        and not rep.contradictions
        and rep.one_page
    )
    return rep


def verify_refusal(
    refusal: dict[str, Any], tool_results: list[dict[str, Any]], question: str
) -> Report:
    """A refusal may repeat numbers from the question (to reject them) but must not add new ones."""
    rel_tol = float(load_config("agent")["verify"]["rel_tol"])
    q_nums = [
        v for m in NUMBER.finditer(strip_references(question)) for v, _ in parse_number(_num(m))
    ]
    candidates = [v for r in tool_results for v in numeric_leaves(r)] + q_nums
    rep = Report()
    rep.numbers, rep.unverified_numbers, _ = check_numbers(
        refusal.get("grund", ""), candidates, set(), rel_tol
    )
    rep.has_citation = True
    rep.passed = not rep.unverified_numbers
    return rep
