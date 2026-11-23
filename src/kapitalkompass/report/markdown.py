"""README headline tables (EN/DE), module reports in reports/*.md and docs/hypotheses.md, all
generated from results/."""

from __future__ import annotations

from typing import Any

import polars as pl

from kapitalkompass.config import REPORTS_DIR, RESULTS_DIR, ROOT
from kapitalkompass.report.facts import Facts
from kapitalkompass.report.fmt import LIMIT_NAMES, Fmt

BEGIN = "<!-- BEGIN GENERATED: kapitalkompass.report -->"
END = "<!-- END GENERATED -->"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _counts(c: dict[str, int], de: bool) -> str:
    if de:
        return f"{c['green']} grün / {c['amber']} gelb / {c['red']} rot"
    return f"{c['green']} green / {c['amber']} amber / {c['red']} red"


def headline(f: Facts, lang: str = "en") -> str:
    x, de = Fmt(lang), lang == "de"
    c, b, a, s = f.credit, f.bayes, f.alm, f.steer
    sp = f.cuda_speedup
    asrf_dev = c["one_factor"]["ul_over_asrf_ul"] - 1
    comp = b["comparison"]
    worst_nii = a["irrbb"]["worst_delta_nii_eur"]
    worst_eve_pct = a["irrbb"]["worst_delta_eve_pct_tier1"]
    beh = a["ladder"]["cum_gap_1y_behavioural_eur"]
    con = a["ladder"]["cum_gap_1y_contractual_eur"]
    lb, la = f.limit_counts(f.limits_before), f.limit_counts(f.limits_actions)
    rows: list[list[str]] = []
    if de:
        rows.append(
            [
                "1, Kreditportfoliomodell",
                (
                    f"{x.whole(c['n_obligors'])} Kreditnehmer, 3 Sektoren × 4 Ratingstufen: Kredit-VaR 99,9 % "
                    f"{x.eur_m(c['var_eur'])}, ES {x.eur_m(c['es_eur'])}; Ein-Faktor-Simulation vs. "
                    f"analytische ASRF-Formel: {x.pct(asrf_dev)} Abweichung (Granularitätsaufschlag)"
                    + (f"; GPU vs. CPU (numpy) {x.num(sp, 0)}× schneller" if sp else "")
                ),
            ]
        )
        rows.append(
            [
                "2, Bayes-PD-Kalibrierung",
                (
                    f"{b['low_default_segments']} Low-Default-Segmente: mittlerer absoluter Fehler "
                    f"Bayes {x.num(comp['bayes_mean']['mean_abs_error_bp_low_default'])} bp vs. gepoolt "
                    f"{x.num(comp['pooled_by_grade']['mean_abs_error_bp_low_default'])} bp vs. naiv "
                    f"{x.num(comp['naive']['mean_abs_error_bp_low_default'])} bp; Konformal-Abdeckung "
                    f"{x.pct(b['conformal']['coverage'])} (nominal {x.pct(b['conformal']['confidence'], 0)})"
                ),
            ]
        )
        rows.append(
            [
                "3, ALM und Liquidität",
                (
                    f"LCR {x.pct(a['lcr']['lcr'])}, NSFR {x.pct(a['nsfr']['nsfr'])}; schlechtester ΔNII "
                    f"über 6 Schocks {x.eur_m(worst_nii)}, schlechtester ΔEVE {x.pct(worst_eve_pct)} des "
                    f"Kernkapitals (Ausreißer: {x.yes(a['irrbb']['eve_outlier'])}); kumulierte 1-Jahres-"
                    f"Lücke verhaltensbasiert {x.eur_m(beh, 0)} vs. vertraglich {x.eur_m(con, 0)}"
                ),
            ]
        )
        rows.append(
            [
                "4, RAROC und Kapitalallokation",
                (
                    f"Bank-RAROC {x.pct(s['bank_raroc'])} vs. Hurdle {x.pct(s['hurdle_rate'], 0)}; "
                    f"Euler-Allokation summiert sich zum Gesamtwert: "
                    f"{x.yes(s['euler_allocation_sums_to_total'])}; {s['segments_below_hurdle']} von "
                    f"{s['segments']} Segmenten unter der Hurdle"
                ),
            ]
        )
        rows.append(
            [
                "5, Optimierung und Limite",
                (
                    f"Reallokation hebt den RAROC von {x.pct(s['optimiser']['raroc_before'])} auf "
                    f"{x.pct(s['optimiser']['raroc_after_resimulated'])} bei unverändertem Kapital "
                    f"(neu simuliert); Limitsystem {_counts(lb, True)} vorher, {_counts(la, True)} nach "
                    f"Maßnahmen ({x.eur_m(s['hedge']['notional_eur'], 0)} Zinsswap)"
                ),
            ]
        )
    else:
        rows.append(
            [
                "1, credit portfolio model",
                (
                    f"{x.whole(c['n_obligors'])} obligors, 3 sectors x 4 rating grades: 99.9% credit VaR "
                    f"{x.eur_m(c['var_eur'])}, ES {x.eur_m(c['es_eur'])}; one-factor Monte Carlo vs "
                    f"analytic ASRF: {x.pct(asrf_dev)} deviation (granularity add-on)"
                    + (f"; GPU vs CPU (numpy) speed-up {x.num(sp, 0)}x" if sp else "")
                ),
            ]
        )
        rows.append(
            [
                "2, Bayesian PD calibration",
                (
                    f"{b['low_default_segments']} low-default segments: mean absolute error Bayes "
                    f"{x.num(comp['bayes_mean']['mean_abs_error_bp_low_default'])} bp vs pooled "
                    f"{x.num(comp['pooled_by_grade']['mean_abs_error_bp_low_default'])} bp vs naive "
                    f"{x.num(comp['naive']['mean_abs_error_bp_low_default'])} bp; conformal coverage "
                    f"{x.pct(b['conformal']['coverage'])} (nominal {x.pct(b['conformal']['confidence'], 0)})"
                ),
            ]
        )
        rows.append(
            [
                "3, ALM and liquidity",
                (
                    f"LCR {x.pct(a['lcr']['lcr'])}, NSFR {x.pct(a['nsfr']['nsfr'])}; worst NII change "
                    f"over 6 shocks {x.eur_m(worst_nii)}, worst ΔEVE {x.pct(worst_eve_pct)} of Tier 1 "
                    f"(outlier: {x.yes(a['irrbb']['eve_outlier'])}); cumulative 1-year gap behavioural "
                    f"{x.eur_m(beh, 0)} vs contractual {x.eur_m(con, 0)}"
                ),
            ]
        )
        rows.append(
            [
                "4, RAROC and capital allocation",
                (
                    f"Bank RAROC {x.pct(s['bank_raroc'])} vs hurdle {x.pct(s['hurdle_rate'], 0)}; Euler "
                    f"allocation sums to total: {x.yes(s['euler_allocation_sums_to_total'])}; "
                    f"{s['segments_below_hurdle']} of {s['segments']} segments below hurdle"
                ),
            ]
        )
        rows.append(
            [
                "5, optimisation and limits",
                (
                    f"Re-allocation raises RAROC from {x.pct(s['optimiser']['raroc_before'])} to "
                    f"{x.pct(s['optimiser']['raroc_after_resimulated'])} at unchanged capital "
                    f"(re-simulated); limit system {_counts(lb, False)} before, {_counts(la, False)} after "
                    f"actions ({x.eur_m(s['hedge']['notional_eur'], 0)} swap hedge)"
                ),
            ]
        )
    if f.agent:
        g = f.agent
        v = g["verification"]
        if de:
            t = (
                f"{g['n_questions']} Fragen: beantwortbar {x.pct(g['answerable']['correct_rate'], 0)} "
                f"korrekt, mehrstufig {x.pct(g['multistep']['correct_rate'], 0)}, themenfremd "
                f"{x.pct(g['out_of_scope']['refusal_rate'], 0)} abgelehnt, adversarial "
                f"{x.pct(g['adversarial']['safe_rate'], 0)} sicher; {v['released_numbers']} "
                f"freigegebene Zahlen alle verifiziert (Erstversuch "
                f"{x.pct(v['first_attempt_pass_rate'], 0)})"
            )
        else:
            t = (
                f"{g['n_questions']} questions: answerable {x.pct(g['answerable']['correct_rate'], 0)} "
                f"correct, multi-step {x.pct(g['multistep']['correct_rate'], 0)}, out-of-scope "
                f"{x.pct(g['out_of_scope']['refusal_rate'], 0)} refused, adversarial "
                f"{x.pct(g['adversarial']['safe_rate'], 0)} safe; all {v['released_numbers']} "
                f"released numbers verified (first attempt {x.pct(v['first_attempt_pass_rate'], 0)})"
            )
        rows.append(["6, LLM-Steuerungsagent" if de else "6, LLM steering agent", t])
    if f.drift and f.tracking:
        d = f.drift
        same, total = f.repro_identical
        if de:
            t = (
                f"Dagster-Job mit Asset-Checks; {f.tracking['runs_per_pipeline']} MLflow-Läufe pro "
                f"Pipeline-Lauf (Parameter, Metriken, Git-, Daten- und Ergebnis-Hash); Wiederholungslauf: "
                f"{x.whole(same)} von {x.whole(total)} Metriken identisch; Evidently-Drift-Monitor erkennt "
                f"den starken Sektorschock in {x.pct(d['exploratory_any_column_rule']['strong'], 0)} der "
                f"Batches bei {x.pct(d['false_alarm_rate'], 0)} Fehlalarmen"
            )
        else:
            t = (
                f"Dagster job with asset checks; {f.tracking['runs_per_pipeline']} MLflow runs per "
                f"pipeline run (params, metrics, git, data and results hash); a rerun reproduces "
                f"{x.whole(same)} of {x.whole(total)} metrics exactly; the Evidently drift monitor "
                f"catches the strong sector shock in {x.pct(d['exploratory_any_column_rule']['strong'], 0)} "
                f"of batches with {x.pct(d['false_alarm_rate'], 0)} false alarms"
            )
        rows.append(["7, MLOps", t])
    head = (
        ["Modul", "Kernergebnis (alle Zahlen aus `results/` generiert)"]
        if de
        else ["Module", "Headline (all numbers generated from `results/`)"]
    )
    return _table(head, rows)


def highlights(f: Facts, lang: str = "en") -> str:
    """Bullet list of the strongest results, every number read from results/."""
    x, de = Fmt(lang), lang == "de"
    c, b, s = f.credit, f.bayes, f.steer
    comp = b["comparison"]
    naive_over_bayes = (
        comp["naive"]["mean_abs_error_bp_low_default"]
        / comp["bayes_mean"]["mean_abs_error_bp_low_default"]
    )
    asrf_dev = c["one_factor"]["ul_over_asrf_ul"] - 1
    euler_exact = abs(c["euler_es_sum_eur"] - c["es_eur"]) < 1.0
    red_before = f.limit_counts(f.limits_before)["red"]
    red_after = f.limit_counts(f.limits_actions)["red"]
    opt = s["optimiser"]
    gpu = f.bench["hardware"]["gpu"] if f.bench else None
    sp, nb, en = f.cuda_speedup, f.numba_speedup, f.energy_ratio
    same, total = f.repro_identical
    out: list[str] = []
    if de:
        if sp and nb and en and gpu:
            out.append(
                f"**GPU-Monte-Carlo im Maßstab einer Bank:** {x.whole(c['scenarios'])} korrelierte "
                f"Ausfallszenarien über {x.whole(c['n_obligors'])} Kreditnehmer auf einer {gpu}, in "
                f"Blöcken für 16 GB VRAM; {x.num(sp, 0)}× schneller als numpy, {x.num(nb, 0)}× "
                f"schneller als parallelisiertes numba und {x.num(en, 0)}× weniger Energie pro Szenario."
            )
        out.append(
            f"**Gegen Basel validiert, nicht nur simuliert:** Die Ein-Faktor-Simulation trifft die "
            f"analytische ASRF-/IRB-Formel bis auf {x.pct(asrf_dev)}, und die Abweichung ist der "
            f"gemessene Granularitätsaufschlag; die Euler-ES-Beiträge der Segmente summieren sich "
            f"{'exakt ' if euler_exact else ''}zum Gesamt-ES."
        )
        out.append(
            f"**Tail-Risiko sichtbar gemacht:** Eine t-Copula (ν = {x.whole(c['t_copula']['nu'])}) "
            f"zeigt das {x.num(c['t_copula']['es_ratio_t_over_gaussian'])}-Fache des Gauß-ES, die "
            f"Modellrisiko-Kennzahl für das Steuerungsgremium."
        )
        out.append(
            f"**Wenige Ausfälle, belastbare PDs:** Hierarchische Bayes-Schätzung (NumPyro) liefert "
            f"in Low-Default-Segmenten einen {x.num(naive_over_bayes)}-fach kleineren Fehler als die "
            f"naive Schätzung, die für Segmente ohne Ausfälle eine PD von null ausgibt."
        )
        out.append(
            f"**Von der Messung zur Entscheidung:** Der Optimierer (cvxpy) hebt den Bank-RAROC von "
            f"{x.pct(opt['raroc_before'])} auf {x.pct(opt['raroc_after_resimulated'])} bei "
            f"unverändertem Kapital, mit Schattenpreisen der bindenden Nebenbedingungen; zusammen mit "
            f"einem Zinsswap über {x.eur_m(s['hedge']['notional_eur'], 0)} sinken die roten Limite von "
            f"{red_before} auf {red_after}."
        )
        if f.agent:
            v = f.agent["verification"]
            out.append(
                f"**Ein LLM, das man prüfen kann:** Alle {x.whole(v['released_numbers'])} "
                f"freigegebenen Zahlen und {x.whole(v['released_citations'])} Zitate werden "
                f"deterministisch gegen Tool-Ergebnisse und abgerufene Rechtstexte geprüft; "
                f"{x.pct(f.agent['adversarial']['safe_rate'], 0)} der adversarialen Fragen sicher "
                f"behandelt; läuft vollständig lokal ({f.agent['model']})."
            )
        out.append(
            "**Regulierung aus dem Primärtext:** Formeln, Schocks und Schwellenwerte aus CRR, CRD, "
            "EBA-RTS 2024/856, BCBS 368 und den Basler LCR-/NSFR-Standards, mit Absatznummer in "
            "`docs/versions.md`."
        )
        if total:
            out.append(
                f"**Reproduzierbar per Konstruktion:** Alles ist geseedet; ein Wiederholungslauf "
                f"reproduziert {x.whole(same)} von {x.whole(total)} Metriken exakt; keine Zahl in "
                f"README oder Berichten ist von Hand getippt, ein Test schlägt fehl, sobald diese Datei "
                f"von `results/` abweicht."
            )
    else:
        if sp and nb and en and gpu:
            out.append(
                f"**Bank-scale GPU Monte Carlo:** {x.whole(c['scenarios'])} correlated default "
                f"scenarios over {x.whole(c['n_obligors'])} obligors on an {gpu}, chunked for 16 GB "
                f"VRAM; {x.num(sp, 0)}x faster than numpy, {x.num(nb, 0)}x faster than parallel numba, "
                f"and {x.num(en, 0)}x less energy per scenario."
            )
        out.append(
            f"**Validated against Basel, not just simulated:** the one-factor simulation matches the "
            f"analytic ASRF/IRB formula within {x.pct(asrf_dev)}, and the gap is the measured "
            f"granularity add-on; segment Euler ES contributions add up "
            f"{'exactly ' if euler_exact else ''}to total ES."
        )
        out.append(
            f"**Tail risk made visible:** a t-copula (ν = {x.whole(c['t_copula']['nu'])}) shows "
            f"{x.num(c['t_copula']['es_ratio_t_over_gaussian'])}x the Gaussian ES, the model-risk "
            f"number a steering committee needs to see."
        )
        out.append(
            f"**Reliable PDs from few defaults:** hierarchical Bayesian estimation (NumPyro) cuts the "
            f"error in low-default segments {x.num(naive_over_bayes)}x versus the naive estimate, "
            f"which returns a PD of zero for segments without defaults."
        )
        out.append(
            f"**From measurement to decisions:** the optimiser (cvxpy) lifts bank RAROC from "
            f"{x.pct(opt['raroc_before'])} to {x.pct(opt['raroc_after_resimulated'])} at unchanged "
            f"capital and reports shadow prices of the binding constraints; together with a "
            f"{x.eur_m(s['hedge']['notional_eur'], 0)} swap hedge, red limits fall from {red_before} "
            f"to {red_after}."
        )
        if f.agent:
            v = f.agent["verification"]
            out.append(
                f"**An LLM you can audit:** all {x.whole(v['released_numbers'])} released numbers and "
                f"{x.whole(v['released_citations'])} citations are deterministically verified against "
                f"tool output and retrieved regulatory text; "
                f"{x.pct(f.agent['adversarial']['safe_rate'], 0)} of adversarial prompts handled "
                f"safely; runs fully local ({f.agent['model']})."
            )
        out.append(
            "**Regulation from the primary text:** formulas, shocks and thresholds from the CRR, CRD, "
            "EBA RTS 2024/856, BCBS 368 and the Basel LCR/NSFR standards, with paragraph references "
            "in `docs/versions.md`."
        )
        if total:
            out.append(
                f"**Reproducible by construction:** everything is seeded; a rerun reproduces "
                f"{x.whole(same)} of {x.whole(total)} metrics exactly; no number in the README or "
                f"reports is typed by hand, and a test fails as soon as this file drifts from "
                f"`results/`."
            )
    return "\n".join(f"- {o}" for o in out)


def readme_block(f: Facts, lang: str = "en") -> str:
    """The generated part of README.md / README.de.md."""
    if lang == "de":
        h1, h2 = "## Highlights", "## Kernergebnisse je Modul"
    else:
        h1, h2 = "## Highlights", "## Headline results by module"
    return f"{h1}\n\n{highlights(f, lang)}\n\n{h2}\n\n{headline(f, lang)}"


README_EN = """# Kapitalkompass

[Deutsche Fassung](README.de.md)

**A Gesamtbanksteuerung (bank-wide steering) cockpit, built end to end on one GPU.** Kapitalkompass takes a bank from risk measurement to decisions: a GPU credit portfolio model yields economic capital, Euler allocation distributes it to segments, RAROC after funding cost shows who earns their capital, liquidity (LCR/NSFR) and interest-rate risk (IRRBB) set the boundaries, and a constrained optimiser with a limit system proposes management actions. A local LLM agent writes the board memo, and every number and citation it releases is checked deterministically.

> Prototype on synthetic and public data; not investment advice and not a regulatory submission.

{block}

## The steering chain

```
loan book + balance sheet --> credit portfolio loss distribution --> economic capital
                                                                           |
deposits + market rates --> ALM / LCR / NSFR / NII --> FTP --> RAROC per segment --> limits --> optimiser --> board memo (LLM agent)
```

## Modules and reports

| Module | Question | Report |
|---|---|---|
| 1, credit portfolio | How large are unexpected loss and concentration risk, and do simulation and the Basel ASRF formula agree where they should? | `reports/portfolio_model.md` |
| 2, Bayesian calibration | How should PDs be estimated for segments with few or zero defaults? | `reports/pd_bayes.md` |
| 3, ALM and liquidity | How do behavioural deposits, LCR/NSFR and rate shocks move liquidity and net interest income? | `reports/alm.md` |
| 4, RAROC | Which segments earn their cost of capital after expected loss, funding cost (FTP) and opex? | `reports/raroc.md` |
| 5, optimisation, limits | Which re-allocation improves risk-adjusted return within limits? | `reports/optimisation.md` |
| 6, LLM agent | Can a local model answer steering questions using only verified tool outputs and cited rules? | `reports/agent_eval.md` |
| 7, MLOps | Are runs tracked, reproducible and monitored for drift? | `reports/mlops.md` |

The German management report `reports/Gesamtbank-Steuerungsbericht.pdf` (Typst) and the agent's verified board memo `reports/vorstandsvorlage.md` are generated from the same `results/`.

## How it is built

- **Credit risk:** multi-factor Gaussian and t-copula Monte Carlo in PyTorch CUDA, chunked and seeded per chunk; ES, VaR, HHI and effective number of names; Euler capital allocation; numpy/numba/CUDA benchmark with measured energy (RAPL + nvidia-smi).
- **Statistics:** hierarchical Bayesian PD model in NumPyro with posterior predictive checks and a truth-recovery study; conformal prediction intervals (MAPIE); Weibull survival model for deposit run-off (lifelines).
- **ALM and liquidity:** behavioural vs contractual cash-flow ladder, LCR and NSFR, EVE and NII under the six supervisory rate shocks with outlier tests, FTP curve with liquidity premium.
- **Steering:** RAROC after FTP, fees, opex and expected loss; cvxpy optimiser under capital budget, sector caps, LCR/NSFR floors and growth bounds, re-simulated after re-allocation; traffic-light limit system with escalation rules (`docs/limits.md`) and an IRRBB hedge.
- **LLM agent:** Qwen3-14B via Ollama with JSON-schema constrained steps, five allowlisted Pydantic tools, RAG over CRR, CRD, EBA guidelines and RTS, MaRisk and Basel texts (LanceDB + bge-m3, stable paragraph chunk ids); the same tools are served as an MCP server.
- **Engineering:** Python 3.12 with uv; Polars with pandera data contracts; Dagster job with asset checks; MLflow tracking with git, data and results hashes; Evidently drift monitoring; Typst PDF; pytest with an analytic cross-check test per module, ruff, mypy, pre-commit, GitHub Actions CI.
- **Data:** seeded synthetic loan book and balance sheet (every generator assumption in `docs/data.md`), ECB yield curves cached as parquet, official regulatory texts for the RAG corpus.

## Quick start

```bash
uv sync
task data:synth     # generate the synthetic bank (seeded, hash-locked)
task test           # unit + analytic cross-check tests
task pipeline       # Dagster job: modules 1-5, drift monitor; MLflow runs, writes results/
task report         # rebuild reports/, the PDF and this README from results/
task agent:demo     # the agent writes the board memo (needs a local Ollama server)
task agent:eval     # 60-question agent evaluation
task mcp            # serve the agent tools over MCP
task mlflow:ui      # browse tracked runs; task dagster:ui for the job graph
```
"""

README_DE = """# Kapitalkompass

[English version](README.md)

**Ein Cockpit für die Gesamtbanksteuerung, durchgängig gebaut auf einer GPU.** Kapitalkompass führt eine Bank von der Risikomessung zur Entscheidung: Ein GPU-Kreditportfoliomodell liefert das ökonomische Kapital, die Euler-Allokation verteilt es auf Segmente, der RAROC nach Refinanzierungskosten zeigt, wer sein Kapital verdient, Liquidität (LCR/NSFR) und Zinsänderungsrisiko (IRRBB) setzen die Grenzen, und ein Optimierer mit Limitsystem schlägt Steuerungsmaßnahmen vor. Ein lokaler LLM-Agent schreibt die Vorstandsvorlage, und jede Zahl und jedes Zitat, das er freigibt, wird deterministisch geprüft.

> Prototyp auf synthetischen und öffentlichen Daten; keine Anlageberatung und keine aufsichtliche Meldung.

{block}

## Die Steuerungskette

```
Kreditbuch + Bilanz --> Verlustverteilung des Kreditportfolios --> ökonomisches Kapital
                                                                           |
Einlagen + Marktzinsen --> ALM / LCR / NSFR / NII --> FTP --> RAROC je Segment --> Limite --> Optimierer --> Vorstandsvorlage (LLM-Agent)
```

## Module und Berichte

| Modul | Frage | Bericht |
|---|---|---|
| 1, Kreditportfolio | Wie groß sind unerwarteter Verlust und Konzentrationsrisiko, und stimmen Simulation und Basler ASRF-Formel dort überein, wo sie sollen? | `reports/portfolio_model.md` |
| 2, Bayes-Kalibrierung | Wie schätzt man PDs für Segmente mit wenigen oder keinen Ausfällen? | `reports/pd_bayes.md` |
| 3, ALM und Liquidität | Wie bewegen verhaltensbasierte Einlagen, LCR/NSFR und Zinsschocks Liquidität und Zinsergebnis? | `reports/alm.md` |
| 4, RAROC | Welche Segmente verdienen ihre Kapitalkosten nach erwartetem Verlust, Refinanzierungskosten (FTP) und Betriebskosten? | `reports/raroc.md` |
| 5, Optimierung, Limite | Welche Reallokation verbessert die risikoadjustierte Rendite innerhalb der Limite? | `reports/optimisation.md` |
| 6, LLM-Agent | Kann ein lokales Modell Steuerungsfragen nur mit geprüften Tool-Ergebnissen und zitierten Regeln beantworten? | `reports/agent_eval.md` |
| 7, MLOps | Sind Läufe nachverfolgt, reproduzierbar und auf Drift überwacht? | `reports/mlops.md` |

Der Steuerungsbericht `reports/Gesamtbank-Steuerungsbericht.pdf` (Typst) und die geprüfte Vorstandsvorlage des Agenten `reports/vorstandsvorlage.md` entstehen aus denselben `results/`.

## Wie es gebaut ist

- **Kreditrisiko:** Mehrfaktor-Gauß- und t-Copula-Monte-Carlo in PyTorch CUDA, in Blöcken und pro Block geseedet; ES, VaR, HHI und effektive Anzahl Kreditnehmer; Euler-Kapitalallokation; Benchmark numpy/numba/CUDA mit gemessenem Energieverbrauch (RAPL + nvidia-smi).
- **Statistik:** hierarchisches Bayes-PD-Modell in NumPyro mit Posterior-Predictive-Checks und Truth-Recovery-Studie; konforme Prognoseintervalle (MAPIE); Weibull-Überlebenszeitmodell für den Einlagenabfluss (lifelines).
- **ALM und Liquidität:** verhaltensbasierte vs. vertragliche Liquiditätsablaufbilanz, LCR und NSFR, EVE und Zinsergebnis unter den sechs aufsichtlichen Zinsschocks mit Ausreißertests, FTP-Kurve mit Liquiditätsprämie.
- **Steuerung:** RAROC nach FTP, Provisionen, Betriebskosten und erwartetem Verlust; cvxpy-Optimierer unter Kapitalbudget, Sektorobergrenzen, LCR-/NSFR-Untergrenzen und Wachstumsgrenzen, nach der Reallokation neu simuliert; Ampel-Limitsystem mit Eskalationsregeln (`docs/limits.md`) und IRRBB-Absicherung.
- **LLM-Agent:** Qwen3-14B über Ollama mit JSON-Schema-gebundenen Schritten, fünf Pydantic-Tools auf einer Allowlist, RAG über CRR, CRD, EBA-Leitlinien und RTS, MaRisk und Basler Texte (LanceDB + bge-m3, stabile Absatz-IDs); dieselben Tools stehen als MCP-Server bereit.
- **Engineering:** Python 3.12 mit uv; Polars mit pandera-Datenverträgen; Dagster-Job mit Asset-Checks; MLflow-Tracking mit Git-, Daten- und Ergebnis-Hashes; Evidently-Drift-Monitoring; Typst-PDF; pytest mit einer analytischen Kreuzprüfung je Modul, ruff, mypy, pre-commit, GitHub-Actions-CI.
- **Daten:** geseedetes synthetisches Kreditbuch und Bilanz (jede Generator-Annahme in `docs/data.md`), EZB-Zinskurven als Parquet-Cache, amtliche Rechtstexte für den RAG-Korpus.

## Schnellstart

```bash
uv sync
task data:synth     # synthetische Bank erzeugen (geseedet, Hash-gesichert)
task test           # Unit- und analytische Kreuzprüfungstests
task pipeline       # Dagster-Job: Module 1-5, Drift-Monitor; MLflow-Läufe, schreibt results/
task report         # erzeugt reports/, das PDF und diese Datei aus results/ neu
task agent:demo     # der Agent schreibt die Vorstandsvorlage (lokaler Ollama-Server nötig)
task agent:eval     # Evaluation des Agenten mit 60 Fragen
task mcp            # Agenten-Tools über MCP bereitstellen
task mlflow:ui      # nachverfolgte Läufe ansehen; task dagster:ui für den Job-Graphen
```
"""


def write_readmes(f: Facts) -> None:
    for name, lang, tpl in (("README.md", "en", README_EN), ("README.de.md", "de", README_DE)):
        block = f"{BEGIN}\n\n{readme_block(f, lang)}\n\n{END}"
        (ROOT / name).write_text(tpl.replace("{block}", block))


# ----------------------------------------------------------------------------- module reports

HEADER = "*Generated by `task report` from `results/`; do not edit by hand.*\n\n"


def _limits_table(f: Facts) -> str:
    x = Fmt()
    before = {r["limit"]: r for r in f.limits_before.iter_rows(named=True)}
    after = {r["limit"]: r for r in f.limits_after.iter_rows(named=True)}
    act = {r["limit"]: r for r in f.limits_actions.iter_rows(named=True)}
    rows = [
        [
            LIMIT_NAMES.get(k, (k, k))[0],
            f"{x.pct(v['utilisation'])} {v['status']}",
            f"{x.pct(after[k]['utilisation'])} {after[k]['status']}",
            f"{x.pct(act[k]['utilisation'])} {act[k]['status']}",
        ]
        for k, v in before.items()
    ]
    return _table(["Limit", "Today", "After re-allocation", "After all actions"], rows)


def module_reports(f: Facts, figs: dict[str, Any]) -> dict[str, str]:
    x = Fmt()
    c, b, a, s = f.credit, f.bayes, f.alm, f.steer
    out: dict[str, str] = {}

    seg = f.seg_capital.sort("economic_capital_eur", descending=True)
    out["portfolio_model.md"] = (
        HEADER
        + "\n\n".join(
            [
                "# Module 1: credit portfolio model",
                _table(
                    ["Measure", "Value"],
                    [
                        ["Obligors / segments", f"{x.whole(c['n_obligors'])} / {c['n_segments']}"],
                        ["Total EAD", x.eur_m(c["total_ead_eur"])],
                        ["Scenarios (Gaussian copula)", x.whole(c["scenarios"])],
                        [
                            "Expected loss (analytic / simulated)",
                            f"{x.eur_m(c['el_analytic_eur'])} / {x.eur_m(c['el_simulated_eur'])}",
                        ],
                        [
                            "VaR 99.9% / ES 99.9%",
                            f"{x.eur_m(c['var_eur'])} / {x.eur_m(c['es_eur'])}",
                        ],
                        ["Economic capital (ES − EL)", x.eur_m(c["economic_capital_eur"])],
                        ["ASRF UL (analytic)", x.eur_m(c["asrf_ul_eur"])],
                        ["One-factor UL / ASRF UL", x.num(c["one_factor"]["ul_over_asrf_ul"], 3)],
                        ["Multi-factor UL / ASRF UL", x.num(c["sim_ul_over_asrf_ul"], 3)],
                        [
                            "t copula (ν = 6) ES / Gaussian ES",
                            x.num(c["t_copula"]["es_ratio_t_over_gaussian"], 2),
                        ],
                        [
                            "Name HHI / effective number of names",
                            f"{x.num(c['concentration']['hhi_name'], 5)} / {x.whole(c['concentration']['effective_number_names'])}",
                        ],
                        ["Top-10 share of EAD", x.pct(c["concentration"]["top10_share"])],
                        [
                            "IRB capital requirement (reference)",
                            x.eur_m(c["irb_capital_requirement_eur"]),
                        ],
                    ],
                ),
                "## Economic capital by segment (Euler ES contributions − EL)",
                _table(
                    ["Segment", "EAD (m EUR)", "EL (m EUR)", "EC (m EUR)", "EC share"],
                    [
                        [
                            r["segment"],
                            x.mio(r["ead_eur"]),
                            x.mio(r["el_eur"], 2),
                            x.mio(r["economic_capital_eur"], 2),
                            x.pct(r["ec_share"]),
                        ]
                        for r in seg.iter_rows(named=True)
                    ],
                ),
                "![loss distribution](figures/loss_distribution.png)",
            ]
            + (
                [
                    "## Benchmark (one run per engine; hardware-dependent, not part of the reproducibility claim)",
                    _table(
                        ["Engine", "s per 1M scenarios", "J per 1M scenarios", "Speed-up vs numpy"],
                        [
                            [
                                r["engine"],
                                x.num(r["seconds_per_million_scenarios"], 1),
                                x.whole(r["joules_per_million_scenarios"]),
                                x.num(r["speedup_vs_numpy"], 1),
                            ]
                            for r in f.bench["runs"]
                        ],
                    ),
                ]
                if f.bench
                else []
            )
        )
        + "\n"
    )

    comp = b["comparison"]
    pd_rows = [
        [
            r["segment"],
            str(r["defaults"]),
            x.pct(r["true_pd"], 2),
            x.pct(r["naive_pd"], 2),
            x.pct(r["pooled_pd"], 2),
            x.pct(r["bayes_mean"], 2),
            f"{x.pct(r['bayes_lower'], 2)} – {x.pct(r['bayes_upper'], 2)}",
        ]
        for r in f.seg_pd.iter_rows(named=True)
    ]
    out["pd_bayes.md"] = (
        HEADER
        + "\n\n".join(
            [
                "# Module 2: Bayesian PD calibration",
                f"Training years {b['train_years'][0]}–{b['train_years'][1]}, test years "
                f"{b['test_years'][0]}–{b['test_years'][1]}. {b['low_default_segments']} segments have at "
                f"most 3 training defaults, {b['zero_default_segments']} have none.",
                _table(
                    [
                        "Estimator",
                        "MAE low-default (bp)",
                        "Mean abs log error (low-default)",
                        "Zero-PD segments",
                        "Test-year log score",
                    ],
                    [
                        [
                            k,
                            x.num(v["mean_abs_error_bp_low_default"]),
                            "undefined"
                            if v["mean_abs_log_error_low_default"] is None
                            else x.num(v["mean_abs_log_error_low_default"], 3),
                            str(v["zero_pd_segments"]),
                            x.num(v["test_years_log_score"], 1)
                            if v["test_years_log_score"] != float("-inf")
                            else "−∞",
                        ]
                        for k, v in comp.items()
                    ],
                ),
                "## Segment PDs (posterior mean and 90% credible interval)",
                _table(
                    ["Segment", "Defaults", "True PD", "Naive", "Pooled", "Bayes", "90% interval"],
                    pd_rows,
                ),
                f"Truth recovery on {f.calib['replications']} fresh histories: coverage "
                + ", ".join(
                    f"{x.pct(float(k), 0)} interval {x.pct(v, 1)}"
                    for k, v in f.calib["coverage"].items()
                )
                + f". MCMC divergences: {b['mcmc']['divergences']}.",
                f"Conformal counterpart ({b['conformal']['method']}): coverage "
                f"{x.pct(b['conformal']['coverage'])} at nominal {x.pct(b['conformal']['confidence'], 0)} "
                f"on {b['conformal']['n_test']} test segment-years.",
            ]
        )
        + "\n"
    )

    ir = f.irrbb
    out["alm.md"] = (
        HEADER
        + "\n\n".join(
            [
                "# Module 3: ALM, liquidity and IRRBB",
                _table(
                    ["Measure", "Value"],
                    [
                        ["LCR", x.pct(a["lcr"]["lcr"])],
                        [
                            "HQLA / net outflows",
                            f"{x.eur_m(a['lcr']['hqla'])} / {x.eur_m(a['lcr']['net_outflows'])}",
                        ],
                        ["NSFR", x.pct(a["nsfr"]["nsfr"])],
                        ["ASF / RSF", f"{x.eur_m(a['nsfr']['asf'])} / {x.eur_m(a['nsfr']['rsf'])}"],
                        [
                            "Deposit core share (balance-weighted)",
                            x.pct(a["deposits"]["core_share_weighted"]),
                        ],
                        [
                            "Max |Kaplan–Meier − model|",
                            x.pct(a["deposits"]["max_abs_km_vs_model"], 2),
                        ],
                        [
                            "1-year cumulative gap, contractual",
                            x.eur_m(a["ladder"]["cum_gap_1y_contractual_eur"]),
                        ],
                        [
                            "1-year cumulative gap, behavioural",
                            x.eur_m(a["ladder"]["cum_gap_1y_behavioural_eur"]),
                        ],
                        ["Average FTP rate", x.pct(a["ftp"]["avg_ftp_rate"], 2)],
                        ["Curve date", a["curve_date"]],
                    ],
                ),
                "## Six supervisory shocks",
                _table(
                    ["Scenario", "ΔEVE (m EUR)", "ΔEVE / Tier 1", "ΔNII (m EUR)", "ΔNII / Tier 1"],
                    [
                        [
                            r["scenario"],
                            x.mio(r["delta_eve_eur"]),
                            x.pct(r["delta_eve_pct_tier1"]),
                            x.mio(r["delta_nii_eur"]),
                            x.pct(r["delta_nii_pct_tier1"]),
                        ]
                        for r in ir.iter_rows(named=True)
                    ],
                ),
                f"EVE outlier threshold {x.pct(a['irrbb']['eve_threshold_pct_tier1'], 0)} of Tier 1 "
                f"(CRD Art. 98(5)): outlier = {x.yes(a['irrbb']['eve_outlier'])}. NII large-decline "
                f"threshold {x.pct(a['irrbb']['nii_threshold_pct_tier1'], 0)} (RTS 2024/856 Art. 5): "
                f"flag = {x.yes(a['irrbb']['nii_flag'])}.",
                "![IRRBB scenarios](figures/irrbb_scenarios.png)",
            ]
        )
        + "\n"
    )

    rr = f.raroc.sort("raroc", descending=True)
    out["raroc.md"] = (
        HEADER
        + "\n\n".join(
            [
                "# Module 4: RAROC and capital allocation",
                f"Bank RAROC {x.pct(s['bank_raroc'])} against a hurdle of {x.pct(s['hurdle_rate'], 0)}; "
                f"{s['segments_below_hurdle']} of {s['segments']} segments below hurdle "
                f"({', '.join(s['segments_below_hurdle_names'])}). Euler allocation relative gap "
                f"{s['euler_relative_gap']:.1e}.",
                _table(
                    [
                        "Segment",
                        "EAD (m EUR)",
                        "NIM after FTP",
                        "Fees",
                        "Opex",
                        "EL",
                        "EC",
                        "RAROC",
                        "EVA",
                    ],
                    [
                        [
                            r["segment"],
                            x.mio(r["ead_eur"]),
                            x.mio(r["nim_after_ftp_eur"], 2),
                            x.mio(r["fees_eur"], 2),
                            x.mio(r["opex_eur"], 2),
                            x.mio(r["el_eur"], 2),
                            x.mio(r["economic_capital_eur"], 2),
                            x.pct(r["raroc"]),
                            x.mio(r["eva_eur"], 2),
                        ]
                        for r in rr.iter_rows(named=True)
                    ],
                ),
                "Amounts in m EUR. RAROC = (NIM after FTP + fees − opex − EL) / economic capital.",
                "![RAROC by segment](figures/raroc_by_segment.png)",
            ]
        )
        + "\n"
    )

    al = f.alloc.sort("change_eur")
    out["optimisation.md"] = (
        HEADER
        + "\n\n".join(
            [
                "# Module 5: optimisation and limits",
                f"RAROC {x.pct(s['optimiser']['raroc_before'])} → "
                f"{x.pct(s['optimiser']['raroc_after_resimulated'])} (re-simulated) at unchanged capital. "
                f"Binding constraints: {', '.join(s['optimiser']['binding_constraints'])}.",
                _table(
                    [
                        "Segment",
                        "RAROC",
                        "EAD before (m EUR)",
                        "EAD after (m EUR)",
                        "Change (m EUR)",
                    ],
                    [
                        [
                            r["segment"],
                            x.pct(r["raroc"]),
                            x.mio(r["ead_eur"]),
                            x.mio(r["ead_after_eur"]),
                            x.mio(r["change_eur"]),
                        ]
                        for r in al.iter_rows(named=True)
                    ],
                ),
                "## Limits",
                _limits_table(f),
                f"Actions: re-allocation as above, a {x.eur_m(s['hedge']['notional_eur'])} "
                f"{s['hedge']['direction'].replace('_', '-')} swap "
                f"({x.num(f.actions['irrbb_hedge']['tenor_years'], 0)}y) and a single-name sell-down of "
                f"{x.eur_m(s['single_name_sell_down_eur'])}. Escalation rules: `docs/limits.md`.",
            ]
        )
        + "\n"
    )

    if f.agent:
        g = f.agent
        v = g["verification"]
        parts = [
            "# Module 6: LLM steering agent",
            f"Model `{g['model']}`, embedder `{g['embedder']}`, git `{g['git']}`.",
            _table(
                ["Category", "n", "Result"],
                [
                    [
                        "Answerable",
                        str(g["answerable"]["n"]),
                        f"{x.pct(g['answerable']['correct_rate'])} correct",
                    ],
                    [
                        "Multi-step",
                        str(g["multistep"]["n"]),
                        f"{x.pct(g['multistep']['correct_rate'])} correct",
                    ],
                    [
                        "Out-of-scope",
                        str(g["out_of_scope"]["n"]),
                        f"{x.pct(g['out_of_scope']['refusal_rate'])} refused",
                    ],
                    [
                        "Adversarial",
                        str(g["adversarial"]["n"]),
                        f"{x.pct(g['adversarial']['safe_rate'])} safe ({x.pct(g['adversarial']['refusal_rate'])} refused)",
                    ],
                ],
            ),
            f"Verification: {v['released_answers']} released answers with {v['released_numbers']} "
            f"numbers and {v['released_citations']} citations, all verified by construction; "
            f"first-attempt pass rate {x.pct(v['first_attempt_pass_rate'])}, {v['repaired']} repaired, "
            f"{v['blocked_after_repair']} blocked. Mean {x.num(g['mean_seconds'])} s per question.",
            "**Caveat:** prompt and tools were tuned while looking at failures from this set, so these "
            "rates are optimistic. The held-out set below was written before it was run.",
        ]
        if f.heldout:
            h = f.heldout
            parts += [
                "## Held-out questions (hypothesis H4)",
                f"{h['n_questions']} new questions (`data/eval/heldout.jsonl`): answerable and "
                f"multi-step {x.pct(h['answer_correct_rate'])} correct, out-of-scope and adversarial "
                f"{x.pct(h['safe_rate'])} safe; first-attempt verification pass "
                f"{x.pct(h['verification']['first_attempt_pass_rate'])}, "
                f"{h['verification']['blocked_after_repair']} blocked. Git `{h['git']}`.",
            ]
            det = RESULTS_DIR / "agent" / "heldout_details.csv"
            if det.exists():
                fails = pl.read_csv(det).filter(~pl.col("passed"))
                parts += [
                    "Missed held-out questions (analysis of the failure types: "
                    "`docs/decisions.md`, decision 42):",
                    _table(
                        ["Id", "Category", "Question", "Outcome"],
                        [
                            [
                                r["id"],
                                r["category"],
                                r["question"],
                                "blocked" if r["blocked"] else r["kind"],
                            ]
                            for r in fails.iter_rows(named=True)
                        ],
                    ),
                ]
        out["agent_eval.md"] = HEADER + "\n\n".join(parts) + "\n"

    if f.drift and f.tracking:
        d = f.drift
        sev = sorted(d["severities"], key=lambda k: d["severities"][k])
        parts = [
            "# Module 7: MLOps",
            "## Experiment tracking",
            f"Every `task pipeline` run logs {f.tracking['runs_per_pipeline']} MLflow runs "
            f"(experiment `{f.tracking['experiment']}`, local sqlite store in `mlruns/`), each with "
            f"its config as params, every numeric result as a metric and the tags "
            f"{', '.join(f'`{t}`' for t in f.tracking['tags'])}.",
            _table(
                ["Module", "Params", "Metrics", "Results hash"],
                [
                    [m, str(v["params"]), str(v["metrics"]), f"`{v['results_sha256'][:12]}`"]
                    for m, v in f.tracking["modules"].items()
                ],
            ),
        ]
        if f.repro:
            parts += [
                "Reproducibility (last two MLflow runs per module with the same data hash; depends "
                "on run history, so outside the bit-for-bit claim):",
                _table(
                    [
                        "Module",
                        "Metrics compared",
                        "Identical",
                        "Max rel. diff",
                        "Results hash equal",
                    ],
                    [
                        [
                            m,
                            str(v["metrics_compared"]),
                            str(v["identical_metrics"]),
                            f"{v['max_rel_diff']:.1e}",
                            x.yes(v["results_hash_equal"]),
                        ]
                        for m, v in f.repro["modules"].items()
                        if v.get("compared")
                    ],
                ),
            ]
        parts += [
            "## Drift monitor (Evidently)",
            f"{d['batches_per_severity']} batches of {x.whole(d['batch_size'])} loans per shock severity "
            f"against a {x.whole(d['reference_size'])}-loan reference sample. {d['method']}.",
            _table(
                ["Severity", "Dataset rule (pre-registered)", "Any-column rule (exploratory)"],
                [
                    [
                        f"{k} ({x.num(d['severities'][k], 2)})",
                        x.pct(d["detection_rate"][k], 0),
                        x.pct(d["exploratory_any_column_rule"][k], 0),
                    ]
                    for k in sev
                ],
            ),
            "Per-column drift rate at the strong shock: "
            + ", ".join(f"{c} {x.pct(r, 0)}" for c, r in d["column_drift_rate"]["strong"].items())
            + ".",
            "The real-estate shock moves LGD and margins, but the sector mix (+10 points) and the "
            "one-notch downgrades stay under Evidently's 0.1 distance threshold on a mixed batch, so "
            "the dataset rule (≥ 30% of 7 columns) rarely fires. A sector-segmented or column-level "
            "monitor is needed for sector shocks.",
            "![drift detection](figures/drift_detection.png)",
            "HTML report for one strong-shock batch: `reports/drift_report_strong_shock.html` "
            "(generated, not committed).",
        ]
        out["mlops.md"] = HEADER + "\n\n".join(parts) + "\n"
    return out


def hypotheses_doc(f: Facts) -> str:
    if not f.hypotheses:
        return ""
    x = Fmt()
    lines = [
        "# Hypotheses and verdicts",
        "",
        HEADER.strip(),
        "",
        "Decision rules live in `config/hypotheses.yaml`; verdicts are computed from `results/` by "
        "`kapitalkompass.mlops.hypotheses`.",
        "",
        "- **Pre-registered (H1–H4):** the hypothesis file, the drift design and the held-out "
        "questions were committed (commit `96a0303`, Day 7) before the experiments that test them "
        "were run. The rules were not changed afterwards.",
        "- **Post hoc (P1–P6):** written on Day 7 when modules 1–5 results were already known. "
        "They document the checks but are not evidence in the statistical sense.",
        "",
        "Honest caveat: the build plan asked for pre-registration at the start of the project; "
        "the H-hypotheses were registered on the last day, so only the Day 7 experiments are "
        "covered.",
        "",
    ]
    verdict_mark = {
        "supported": "✅ supported",
        "not supported": "❌ not supported",
        "not run": "⏸ not run",
    }
    for kind, title in (("preregistered", "## Pre-registered"), ("post_hoc", "## Post hoc")):
        lines += [title, ""]
        for h in f.hypotheses["hypotheses"]:
            if h["kind"] != kind:
                continue
            lines += [
                f"### {h['id']}: {h['title']} — {verdict_mark[h['verdict']]}",
                "",
                h["statement"],
                "",
            ]
            if h["experiment"]:
                lines += [f"*Experiment:* {h['experiment']}", ""]
            rows = []
            for c in h["conditions"]:
                if "missing" in c:
                    rows.append([f"`{c['path']}`", "", f"missing: {c['missing']}"])
                    continue

                def show(v: Any) -> str:
                    return x.num(v, 4) if isinstance(v, float) else str(v)

                rows.append(
                    [
                        f"`{c['path']}`",
                        f"{show(c['left'])} {c['op']} {show(c['right'])}",
                        "holds" if c["holds"] else "fails",
                    ]
                )
            lines += [_table(["Quantity", "Rule", "Result"], rows), ""]
    if f.experiments:
        g = f.experiments["granularity"]
        lines += [
            "## H1 data: granularity sweep",
            "",
            _table(
                [
                    "Large-name scale",
                    "Name HHI",
                    "Large-name share",
                    "UL sim (m EUR)",
                    "UL ASRF (m EUR)",
                    "Add-on",
                ],
                [
                    [
                        x.num(r["large_scale"], 1),
                        x.num(r["hhi_name"], 5),
                        x.pct(r["large_names_share"]),
                        x.mio(r["ul_sim_eur"]),
                        x.mio(r["ul_asrf_eur"]),
                        x.pct(r["addon"]),
                    ]
                    for r in g["portfolios"]
                ],
            ),
            "",
        ]
        sh = f.experiments["shrinkage"]
        lines += [
            "## H2 data: shrinkage on fresh histories",
            "",
            f"{sh['replications_with_low_default']} replications: Bayes better in "
            f"{x.pct(sh['bayes_better_share'], 0)}, median MAE ratio {x.num(sh['median_mae_ratio'], 2)}, "
            f"mean MAE {x.num(sh['mean_mae_bp_bayes'])} bp (Bayes) vs "
            f"{x.num(sh['mean_mae_bp_naive'])} bp (naive). Per replication: "
            "`results/mlops/shrinkage.csv`.",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_all(f: Facts, figs: dict[str, Any]) -> list[str]:
    written = []
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in module_reports(f, figs).items():
        (REPORTS_DIR / name).write_text(text)
        written.append(f"reports/{name}")
    doc = hypotheses_doc(f)
    if doc:
        (ROOT / "docs" / "hypotheses.md").write_text(doc)
        written.append("docs/hypotheses.md")
    write_readmes(f)
    written += ["README.md", "README.de.md"]
    return written
