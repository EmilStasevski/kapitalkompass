# Kapitalkompass

[English version](README.md)

Ein Cockpit für die Gesamtbanksteuerung einer stilisierten Bank: Wo wird Kapital gebunden, was verdient es, ist es liquiditätsseitig refinanzierbar, und welche Steuerungsmaßnahme verbessert das Bild? Das Projekt umfasst Kreditportfoliorisiko mit Konzentrationen, bayesianische PD-Kalibrierung für Low-Default-Portfolios, ALM und Liquidität (LCR/NSFR, IRRBB/NII), RAROC und Kapitalallokation, eine Optimierung unter Nebenbedingungen mit Limitsystem und einen lokalen LLM-Agenten, der Steuerungsfragen über geprüfte Tool-Aufrufe beantwortet. Alles läuft lokal auf einer 16-GB-GPU. Jede Zahl in dieser Datei wird von `task report` aus `results/` erzeugt.

> **Keine Anlageberatung, keine aufsichtliche Meldung.** Prototyp auf synthetischen und öffentlichen Daten. Aufsichtsrechtliche Formeln wurden gegen den Primärtext geprüft (`docs/versions.md`).

## Kernergebnisse

<!-- BEGIN GENERATED: kapitalkompass.report -->

| Modul | Kernergebnis (alle Zahlen aus `results/` generiert) |
|---|---|
| 1, Kreditportfoliomodell | 50.000 Kreditnehmer, 3 Sektoren × 4 Ratingstufen: Kredit-VaR 99,9 % 203,9 Mio. EUR, ES 242,3 Mio. EUR; Ein-Faktor-Simulation vs. analytische ASRF-Formel: 2,2 % Abweichung (Granularitätsaufschlag); GPU vs. CPU (numpy) 177× schneller |
| 2, Bayes-PD-Kalibrierung | 4 Low-Default-Segmente: mittlerer absoluter Fehler Bayes 7,1 bp vs. gepoolt 13,7 bp vs. naiv 15,1 bp; Konformal-Abdeckung 88,9 % (nominal 90 %) |
| 3, ALM und Liquidität | LCR 239,3 %, NSFR 119,1 %; schlechtester ΔNII über 6 Schocks −13,1 Mio. EUR, schlechtester ΔEVE −21,3 % des Kernkapitals (Ausreißer: ja); kumulierte 1-Jahres-Lücke verhaltensbasiert 295 Mio. EUR vs. vertraglich −3.368 Mio. EUR |
| 4, RAROC und Kapitalallokation | Bank-RAROC 15,9 % vs. Hurdle 10 %; Euler-Allokation summiert sich zum Gesamtwert: ja; 3 von 12 Segmenten unter der Hurdle |
| 5, Optimierung und Limite | Reallokation hebt den RAROC von 15,9 % auf 17,0 % bei unverändertem Kapital (neu simuliert); Limitsystem 4 grün / 1 gelb / 2 rot vorher, 4 grün / 3 gelb / 0 rot nach Maßnahmen (599 Mio. EUR Zinsswap) |
| 6, LLM-Steuerungsagent | 60 Fragen: beantwortbar 100 % korrekt, mehrstufig 92 %, themenfremd 100 % abgelehnt, adversarial 100 % sicher; 211 freigegebene Zahlen alle verifiziert (Erstversuch 92 %); Hold-out (20 neue Fragen): 71 % korrekt, 100 % sicher |
| 7, MLOps | 7 MLflow-Läufe pro Pipeline-Lauf (Parameter, Metriken, Git-, Daten- und Ergebnis-Hash); Drift-Monitor: vorregistrierte Datensatz-Regel erkennt 4 von 60 geschockten Batches, Einzelspalten-Regel 100 % beim starken Schock, Fehlalarme 0 %; 2 von 4 vorregistrierten Hypothesen gestützt |

<!-- END GENERATED -->

## Berichte

- `reports/Gesamtbank-Steuerungsbericht.pdf`: Steuerungsbericht (Typst), aus `results/` erzeugt
- `reports/vorstandsvorlage.md`: vom Agenten erstellte und deterministisch geprüfte Vorstandsvorlage
- `reports/*.md`: ein Bericht je Modul; `docs/hypotheses.md`: Hypothesen und Urteile

## Schnellstart

```bash
uv sync
task test           # Unit- und analytische Kreuzprüfungstests
task pipeline       # Dagster-Job: Module 1-5 + Drift-Monitor + Experimente, schreibt results/
task report         # erzeugt reports/, PDF und die Tabellen oben neu
task agent:eval     # Evaluation von Modul 6 (lokaler LLM-Server nötig)
```
