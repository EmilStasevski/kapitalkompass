"""German PDF "Gesamtbank-Steuerungsbericht" (Typst), generated from results/ only."""

from __future__ import annotations

from typing import Any

from kapitalkompass.config import REPORTS_DIR, load_config
from kapitalkompass.report.facts import Facts
from kapitalkompass.report.fmt import LIMIT_NAMES, SCENARIO_DE, STATUS_DE, Fmt

PDF_NAME = "Gesamtbank-Steuerungsbericht.pdf"
TYP_NAME = "steuerungsbericht.typ"


def esc(s: str) -> str:
    """Escape text for Typst markup."""
    for ch in "\\#$*_@<>`[]~":
        s = s.replace(ch, "\\" + ch)
    return s


def table(header: list[str], rows: list[list[str]], align: str = "") -> str:
    n = len(header)
    al = align or "(left," + ", ".join(["right"] * (n - 1)) + ")"
    cells = [f"[*{esc(h)}*]" for h in header]
    for r in rows:
        cells += [f"[{esc(c)}]" for c in r]
    return (
        f"#table(columns: {n}, align: {al}, stroke: none, inset: (x: 5pt, y: 3pt),\n"
        '  fill: (_, y) => if y == 0 { rgb("#ecebe7") },\n'
        "  table.hline(y: 1, stroke: 0.5pt),\n  " + ", ".join(cells) + ")\n"
    )


def figure(path: str, caption: str) -> str:
    return f'#figure(image("{path}", width: 100%), caption: [{esc(caption)}])\n'


def build_typ(f: Facts, figs: dict[str, Any]) -> str:
    x = Fmt("de")
    c, b, a, s = f.credit, f.bayes, f.alm, f.steer
    asof = load_config("bank")["reporting_date"]
    lb, la = f.limit_counts(f.limits_before), f.limit_counts(f.limits_actions)
    out: list[str] = [
        '#set document(title: "Gesamtbank-Steuerungsbericht", author: "Kapitalkompass")',
        '#set page(paper: "a4", margin: (x: 2cm, y: 2cm), numbering: "1 / 1",',
        '  header: align(right, text(8pt, fill: rgb("#52514e"))[Kapitalkompass · synthetische Beispielbank]))',
        '#set text(lang: "de", size: 10pt)',
        "#set par(justify: true)",
        '#set heading(numbering: "1.")',
        '#show figure.caption: set text(8.5pt, fill: rgb("#52514e"))',
        "",
        '#align(left)[#text(20pt, weight: "bold")[Gesamtbank-Steuerungsbericht]]',
        f"#text(11pt)[Stichtag {esc(asof)} · Zinskurve vom {esc(a['curve_date'])} · "
        "alle Zahlen automatisch aus results/ erzeugt]",
        "",
        '#block(fill: rgb("#f5f4f0"), inset: 8pt, radius: 3pt, width: 100%)[#text(9pt)['
        + esc(
            "Prototyp auf einer synthetischen Bank mit öffentlichen Zinskurven; keine "
            "aufsichtliche Meldung und keine Anlageberatung. Aufsichtsrechtliche Formeln sind "
            "gegen den Primärtext geprüft (docs/versions.md)."
        )
        + "]]",
        "",
        "= Zusammenfassung",
        table(
            ["Kennzahl", "Wert", "Einordnung"],
            [
                ["Bank-RAROC", x.pct(s["bank_raroc"]), f"Hurdle {x.pct(s['hurdle_rate'], 0)}"],
                [
                    "Ökonomisches Kapital (Kredit)",
                    x.eur_m(c["economic_capital_eur"]),
                    "ES 99,9 % − EL",
                ],
                [
                    "Kredit-VaR / ES 99,9 %",
                    f"{x.eur_m(c['var_eur'])} / {x.eur_m(c['es_eur'])}",
                    f"{x.whole(c['scenarios'])} Szenarien",
                ],
                [
                    "LCR / NSFR",
                    f"{x.pct(a['lcr']['lcr'])} / {x.pct(a['nsfr']['nsfr'])}",
                    "Minimum je 100 %",
                ],
                [
                    "Schlechtester ΔEVE",
                    x.pct(a["irrbb"]["worst_delta_eve_pct_tier1"]),
                    f"des Kernkapitals ({SCENARIO_DE[a['irrbb']['worst_eve_scenario']]}); "
                    f"Schwelle {x.pct(a['irrbb']['eve_threshold_pct_tier1'], 0)}",
                ],
                [
                    "Limite heute",
                    f"{lb['green']} / {lb['amber']} / {lb['red']}",
                    "grün / gelb / rot",
                ],
                [
                    "Limite nach Maßnahmen",
                    f"{la['green']} / {la['amber']} / {la['red']}",
                    "grün / gelb / rot",
                ],
                [
                    "RAROC nach Reallokation",
                    x.pct(s["optimiser"]["raroc_after_resimulated"]),
                    "bei unverändertem Kapital, neu simuliert",
                ],
            ],
            "(left, right, left)",
        ),
        esc(
            f"Die Bank verdient insgesamt über der Hurdle Rate, aber {s['segments_below_hurdle']} von "
            f"{s['segments']} Segmenten verfehlen sie ({', '.join(s['segments_below_hurdle_names'])}). "
            f"Rote Limite heute: {', '.join(LIMIT_NAMES[k][1] for k in s['red_before'])}"
            + (
                " (die Bank ist im EVE-Ausreißertest auffällig)"
                if a["irrbb"]["eve_outlier"]
                else ""
            )
            + ". Empfohlen werden ein "
            f"Receiver-Swap über {x.eur_m(s['hedge']['notional_eur'], 0)}, ein Abbau von "
            f"{x.eur_m(s['single_name_sell_down_eur'])} beim größten Einzelnamen und die "
            f"Reallokation aus Abschnitt 5; danach ist kein Limit mehr rot."
        )
        if la["red"] == 0 and s["bank_raroc"] > s["hurdle_rate"]
        else "",
        "",
        "= Kreditportfolio und ökonomisches Kapital",
        esc(
            f"{x.whole(c['n_obligors'])} Kreditnehmer mit {x.eur_m(c['total_ead_eur'], 0)} EAD in "
            f"{c['n_segments']} Segmenten. Multi-Faktor-Gauß-Copula auf der GPU; der erwartete Verlust "
            f"beträgt {x.eur_m(c['el_analytic_eur'])} (simuliert {x.eur_m(c['el_simulated_eur'])}). "
            f"Im Ein-Faktor-Modell liegt der unerwartete Verlust {x.pct(c['one_factor']['ul_over_asrf_ul'] - 1)} "
            f"über der ASRF-Formel (Namenskonzentration); die Sektordiversifikation senkt ihn auf "
            f"{x.pct(c['sim_ul_over_asrf_ul'], 0)} des ASRF-Werts. Mit t-Copula (ν = 6) steigt der ES "
            f"auf das {x.num(c['t_copula']['es_ratio_t_over_gaussian'], 1)}-Fache."
        ),
        figure(
            "figures/loss_distribution.png", "Verlustverteilung (Gauß-Copula) mit VaR und ES 99,9 %"
        ),
        "= Rentabilität: RAROC je Segment",
        esc(
            "RAROC = (Zinsmarge nach FTP + Provisionen − Betriebskosten − erwarteter Verlust) / "
            "ökonomisches Kapital (Euler-Beiträge zum ES)."
        ),
        figure("figures/raroc_by_segment.png", "RAROC je Segment gegen die Hurdle Rate"),
        table(
            ["Segment", "EAD", "ÖK", "RAROC", "EVA"],
            [
                [
                    r["segment"],
                    x.mio(r["ead_eur"]),
                    x.mio(r["economic_capital_eur"], 1),
                    x.pct(r["raroc"]),
                    x.mio(r["eva_eur"], 2),
                ]
                for r in f.raroc.sort("raroc", descending=True).iter_rows(named=True)
            ],
        ),
        esc("Beträge in Mio. EUR."),
        "",
        "= Liquidität und Zinsänderungsrisiko",
        esc(
            f"LCR {x.pct(a['lcr']['lcr'])} (HQLA {x.eur_m(a['lcr']['hqla'])}, Nettoabflüsse "
            f"{x.eur_m(a['lcr']['net_outflows'])}), NSFR {x.pct(a['nsfr']['nsfr'])}. Das "
            f"Überlebensmodell für Einlagen ergibt einen Bodensatz von "
            f"{x.pct(a['deposits']['core_share_weighted'])}; die kumulierte Liquiditätslücke nach "
            f"einem Jahr ist verhaltensbasiert {x.eur_m(a['ladder']['cum_gap_1y_behavioural_eur'], 0)} "
            f"statt vertraglich {x.eur_m(a['ladder']['cum_gap_1y_contractual_eur'], 0)}."
        ),
        figure(
            "figures/irrbb_scenarios.png",
            "ΔEVE und ΔNII unter den sechs aufsichtlichen Zinsschocks (gestrichelt: Schwellen)",
        ),
        table(
            ["Szenario", "ΔEVE Mio. EUR", "ΔEVE / KK", "ΔNII Mio. EUR", "ΔNII / KK"],
            [
                [
                    SCENARIO_DE[r["scenario"]],
                    x.mio(r["delta_eve_eur"]),
                    x.pct(r["delta_eve_pct_tier1"]),
                    x.mio(r["delta_nii_eur"]),
                    x.pct(r["delta_nii_pct_tier1"]),
                ]
                for r in f.irrbb.iter_rows(named=True)
            ],
        ),
        "= Limite und Maßnahmen",
        table(
            ["Limit", "Heute", "Nach Reallokation", "Nach allen Maßnahmen"],
            [
                [LIMIT_NAMES.get(k, (k, k))[1]]
                + [f"{x.pct(r['utilisation'])} {STATUS_DE[r['status']]}" for r in rows]
                for k, *rows in zip(
                    f.limits_before["limit"],
                    f.limits_before.iter_rows(named=True),
                    f.limits_after.iter_rows(named=True),
                    f.limits_actions.iter_rows(named=True),
                    strict=True,
                )
            ],
        ),
        esc(
            f"Die Optimierung (cvxpy, EVA-Maximierung unter Kapitalbudget, Sektorgrenzen, "
            f"Wachstumsgrenzen, LCR- und NSFR-Untergrenzen) hebt den RAROC von "
            f"{x.pct(s['optimiser']['raroc_before'])} auf {x.pct(s['optimiser']['raroc_after_resimulated'])}. "
            f"Bindend: {', '.join(s['optimiser']['binding_constraints'])}. Ausbau vor allem in "
            f"{', '.join(f.actions['reallocation']['largest_increases'])}, Abbau in "
            f"{', '.join(f.actions['reallocation']['largest_decreases'])}."
        ),
        "",
        "= PD-Kalibrierung für Low-Default-Segmente",
        esc(
            f"Hierarchisches Bayes-Modell (NumPyro) mit Sektorprior und Konjunktureffekt. Auf den "
            f"{b['low_default_segments']} Segmenten mit höchstens drei Ausfällen beträgt der mittlere "
            f"absolute Fehler gegenüber der wahren PD "
            f"{x.num(b['comparison']['bayes_mean']['mean_abs_error_bp_low_default'])} bp (Bayes) gegenüber "
            f"{x.num(b['comparison']['naive']['mean_abs_error_bp_low_default'])} bp (naiv) und "
            f"{x.num(b['comparison']['pooled_by_grade']['mean_abs_error_bp_low_default'])} bp (gepoolt). "
            f"Konformale Intervalle (MAPIE) erreichen {x.pct(b['conformal']['coverage'])} Abdeckung bei "
            f"nominal {x.pct(b['conformal']['confidence'], 0)}."
        ),
        "",
    ]
    if f.agent:
        g, v = f.agent, f.agent["verification"]
        ho = f.heldout
        out += [
            "= Steuerungsagent (lokales LLM)",
            esc(
                f"Der Agent ({g['model']}) beantwortet Steuerungsfragen nur über fünf freigegebene "
                f"Werkzeuge und zitiert Regeltexte aus einem lokalen Index. Jede Zahl und jedes Zitat "
                f"wird vor der Freigabe deterministisch geprüft. Evaluation mit {g['n_questions']} "
                f"Fragen: {x.pct(g['answerable']['correct_rate'], 0)} der beantwortbaren und "
                f"{x.pct(g['multistep']['correct_rate'], 0)} der mehrstufigen Fragen korrekt, "
                f"{x.pct(g['out_of_scope']['refusal_rate'], 0)} der themenfremden abgelehnt, "
                f"{x.pct(g['adversarial']['safe_rate'], 0)} der manipulativen sicher behandelt; "
                f"{v['released_numbers']} freigegebene Zahlen, alle verifiziert. Prompt und Werkzeuge "
                f"wurden an diesem Fragensatz abgestimmt."
            )
            + (
                esc(
                    f" Auf {ho['n_questions']} vorab festgelegten neuen Fragen: "
                    f"{x.pct(ho['answer_correct_rate'], 0)} korrekt, {x.pct(ho['safe_rate'], 0)} sicher."
                )
                if ho
                else ""
            ),
            "",
        ]
    if f.drift and f.hypotheses:
        d = f.drift
        out += [
            "= Betrieb, Monitoring und Hypothesen",
            esc(
                "Jeder Pipeline-Lauf (Dagster-Job mit Asset-Checks) protokolliert je Modul einen "
                "MLflow-Lauf mit Parametern, Metriken, Git-, Daten- und Ergebnis-Hash. Der "
                "Drift-Monitor (Evidently) vergleicht Neugeschäfts-Batches mit dem Bestand; ein "
                f"Immobilienschock wird von der vorab festgelegten Datensatz-Regel nur in "
                f"{d['shocked_batches_flagged']} von {d['shocked_batches']} Batches erkannt, von einer "
                f"Einzelspalten-Regel beim starken Schock in "
                f"{x.pct(d['exploratory_any_column_rule']['strong'], 0)}, ohne Fehlalarme "
                f"({x.pct(d['false_alarm_rate'], 0)})."
            ),
        ]
        if figs.get("drift"):
            out.append(figure("figures/drift_detection.png", "Erkennungsrate nach Schockstärke"))
        verdict = {
            "supported": "gestützt",
            "not supported": "nicht gestützt",
            "not run": "nicht gelaufen",
        }
        kind = {"preregistered": "vorab", "post_hoc": "nachträglich"}
        out.append(
            table(
                ["ID", "Hypothese", "Art", "Urteil"],
                [
                    [h["id"], h["title"], kind[h["kind"]], verdict[h["verdict"]]]
                    for h in f.hypotheses["hypotheses"]
                ],
                "(left, left, left, left)",
            )
        )
        out.append(esc("Hypothesen in englischer Originalfassung; Details in docs/hypotheses.md."))
    out += [
        "",
        '#v(1fr)\n#text(8pt, fill: rgb("#52514e"))['
        + esc(
            "Erzeugt mit task report aus results/. Methodik und Annahmen: docs/decisions.md, "
            "docs/data.md, docs/versions.md, docs/limits.md."
        )
        + "]",
    ]
    return "\n".join(out) + "\n"


def build_pdf(f: Facts, figs: dict[str, Any]) -> str:
    import typst

    src = REPORTS_DIR / TYP_NAME
    src.write_text(build_typ(f, figs))
    typst.compile(str(src), output=str(REPORTS_DIR / PDF_NAME), root=str(REPORTS_DIR))
    return f"reports/{PDF_NAME}"
