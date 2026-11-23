# Kapitalkompass

[English version](README.md)

**Ein Cockpit für die Gesamtbanksteuerung, durchgängig gebaut auf einer GPU.** Kapitalkompass führt eine Bank von der Risikomessung zur Entscheidung: Ein GPU-Kreditportfoliomodell liefert das ökonomische Kapital, die Euler-Allokation verteilt es auf Segmente, der RAROC nach Refinanzierungskosten zeigt, wer sein Kapital verdient, Liquidität (LCR/NSFR) und Zinsänderungsrisiko (IRRBB) setzen die Grenzen, und ein Optimierer mit Limitsystem schlägt Steuerungsmaßnahmen vor. Ein lokaler LLM-Agent schreibt die Vorstandsvorlage, und jede Zahl und jedes Zitat, das er freigibt, wird deterministisch geprüft.

> Prototyp auf synthetischen und öffentlichen Daten; keine Anlageberatung und keine aufsichtliche Meldung.

<!-- BEGIN GENERATED: kapitalkompass.report -->

## Highlights

- **GPU-Monte-Carlo im Maßstab einer Bank:** 2.000.000 korrelierte Ausfallszenarien über 50.000 Kreditnehmer auf einer NVIDIA GeForce RTX 5060 Ti, in Blöcken für 16 GB VRAM; 177× schneller als numpy, 26× schneller als parallelisiertes numba und 74× weniger Energie pro Szenario.
- **Gegen Basel validiert, nicht nur simuliert:** Die Ein-Faktor-Simulation trifft die analytische ASRF-/IRB-Formel bis auf 2,2 %, und die Abweichung ist der gemessene Granularitätsaufschlag; die Euler-ES-Beiträge der Segmente summieren sich exakt zum Gesamt-ES.
- **Tail-Risiko sichtbar gemacht:** Eine t-Copula (ν = 6) zeigt das 3,0-Fache des Gauß-ES, die Modellrisiko-Kennzahl für das Steuerungsgremium.
- **Wenige Ausfälle, belastbare PDs:** Hierarchische Bayes-Schätzung (NumPyro) liefert in Low-Default-Segmenten einen 2,1-fach kleineren Fehler als die naive Schätzung, die für Segmente ohne Ausfälle eine PD von null ausgibt.
- **Von der Messung zur Entscheidung:** Der Optimierer (cvxpy) hebt den Bank-RAROC von 15,9 % auf 17,0 % bei unverändertem Kapital, mit Schattenpreisen der bindenden Nebenbedingungen; zusammen mit einem Zinsswap über 599 Mio. EUR sinken die roten Limite von 2 auf 0.
- **Ein LLM, das man prüfen kann:** Alle 211 freigegebenen Zahlen und 93 Zitate werden deterministisch gegen Tool-Ergebnisse und abgerufene Rechtstexte geprüft; 100 % der adversarialen Fragen sicher behandelt; läuft vollständig lokal (qwen3:14b).
- **Regulierung aus dem Primärtext:** Formeln, Schocks und Schwellenwerte aus CRR, CRD, EBA-RTS 2024/856, BCBS 368 und den Basler LCR-/NSFR-Standards, mit Absatznummer in `docs/versions.md`.
- **Reproduzierbar per Konstruktion:** Alles ist geseedet; ein Wiederholungslauf reproduziert 241 von 241 Metriken exakt; keine Zahl in README oder Berichten ist von Hand getippt, ein Test schlägt fehl, sobald diese Datei von `results/` abweicht.

## Kernergebnisse je Modul

| Modul | Kernergebnis (alle Zahlen aus `results/` generiert) |
|---|---|
| 1, Kreditportfoliomodell | 50.000 Kreditnehmer, 3 Sektoren × 4 Ratingstufen: Kredit-VaR 99,9 % 203,9 Mio. EUR, ES 242,3 Mio. EUR; Ein-Faktor-Simulation vs. analytische ASRF-Formel: 2,2 % Abweichung (Granularitätsaufschlag); GPU vs. CPU (numpy) 177× schneller |
| 2, Bayes-PD-Kalibrierung | 4 Low-Default-Segmente: mittlerer absoluter Fehler Bayes 7,1 bp vs. gepoolt 13,7 bp vs. naiv 15,1 bp; Konformal-Abdeckung 88,9 % (nominal 90 %) |
| 3, ALM und Liquidität | LCR 239,3 %, NSFR 119,1 %; schlechtester ΔNII über 6 Schocks −13,1 Mio. EUR, schlechtester ΔEVE −21,3 % des Kernkapitals (Ausreißer: ja); kumulierte 1-Jahres-Lücke verhaltensbasiert 295 Mio. EUR vs. vertraglich −3.368 Mio. EUR |
| 4, RAROC und Kapitalallokation | Bank-RAROC 15,9 % vs. Hurdle 10 %; Euler-Allokation summiert sich zum Gesamtwert: ja; 3 von 12 Segmenten unter der Hurdle |
| 5, Optimierung und Limite | Reallokation hebt den RAROC von 15,9 % auf 17,0 % bei unverändertem Kapital (neu simuliert); Limitsystem 4 grün / 1 gelb / 2 rot vorher, 4 grün / 3 gelb / 0 rot nach Maßnahmen (599 Mio. EUR Zinsswap) |
| 6, LLM-Steuerungsagent | 60 Fragen: beantwortbar 100 % korrekt, mehrstufig 92 %, themenfremd 100 % abgelehnt, adversarial 100 % sicher; 211 freigegebene Zahlen alle verifiziert (Erstversuch 92 %) |
| 7, MLOps | Dagster-Job mit Asset-Checks; 7 MLflow-Läufe pro Pipeline-Lauf (Parameter, Metriken, Git-, Daten- und Ergebnis-Hash); Wiederholungslauf: 241 von 241 Metriken identisch; Evidently-Drift-Monitor erkennt den starken Sektorschock in 100 % der Batches bei 0 % Fehlalarmen |

<!-- END GENERATED -->

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
