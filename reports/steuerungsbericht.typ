#set document(title: "Gesamtbank-Steuerungsbericht", author: "Kapitalkompass")
#set page(paper: "a4", margin: (x: 2cm, y: 2cm), numbering: "1 / 1",
  header: align(right, text(8pt, fill: rgb("#52514e"))[Kapitalkompass · synthetische Beispielbank]))
#set text(lang: "de", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show figure.caption: set text(8.5pt, fill: rgb("#52514e"))

#align(left)[#text(20pt, weight: "bold")[Gesamtbank-Steuerungsbericht]]
#text(11pt)[Stichtag 2026-11-20 · Zinskurve vom 2026-11-19 · alle Zahlen automatisch aus results/ erzeugt]

#block(fill: rgb("#f5f4f0"), inset: 8pt, radius: 3pt, width: 100%)[#text(9pt)[Prototyp auf einer synthetischen Bank mit öffentlichen Zinskurven; keine aufsichtliche Meldung und keine Anlageberatung. Aufsichtsrechtliche Formeln sind gegen den Primärtext geprüft (docs/versions.md).]]

= Zusammenfassung
#table(columns: 3, align: (left, right, left), stroke: none, inset: (x: 5pt, y: 3pt),
  fill: (_, y) => if y == 0 { rgb("#ecebe7") },
  table.hline(y: 1, stroke: 0.5pt),
  [*Kennzahl*], [*Wert*], [*Einordnung*], [Bank-RAROC], [15,9 %], [Hurdle 10 %], [Ökonomisches Kapital (Kredit)], [214,7 Mio. EUR], [ES 99,9 % − EL], [Kredit-VaR / ES 99,9 %], [203,9 Mio. EUR / 242,3 Mio. EUR], [2.000.000 Szenarien], [LCR / NSFR], [239,3 % / 119,1 %], [Minimum je 100 %], [Schlechtester ΔEVE], [−21,3 %], [des Kernkapitals (Parallel abwärts); Schwelle 15 %], [Limite heute], [4 / 1 / 2], [grün / gelb / rot], [Limite nach Maßnahmen], [4 / 3 / 0], [grün / gelb / rot], [RAROC nach Reallokation], [17,0 %], [bei unverändertem Kapital, neu simuliert])

Die Bank verdient insgesamt über der Hurdle Rate, aber 3 von 12 Segmenten verfehlen sie (manufacturing\_1, manufacturing\_4, trade\_4). Rote Limite heute: Zinsrisiko ΔEVE, Einzelkredit (die Bank ist im EVE-Ausreißertest auffällig). Empfohlen werden ein Receiver-Swap über 599 Mio. EUR, ein Abbau von 11,3 Mio. EUR beim größten Einzelnamen und die Reallokation aus Abschnitt 5; danach ist kein Limit mehr rot.

= Kreditportfolio und ökonomisches Kapital
50.000 Kreditnehmer mit 5.806 Mio. EUR EAD in 12 Segmenten. Multi-Faktor-Gauß-Copula auf der GPU; der erwartete Verlust beträgt 27,6 Mio. EUR (simuliert 27,6 Mio. EUR). Im Ein-Faktor-Modell liegt der unerwartete Verlust 2,2 % über der ASRF-Formel (Namenskonzentration); die Sektordiversifikation senkt ihn auf 77 % des ASRF-Werts. Mit t-Copula (ν = 6) steigt der ES auf das 3,0-Fache.
#figure(image("figures/loss_distribution.png", width: 100%), caption: [Verlustverteilung (Gauß-Copula) mit VaR und ES 99,9 %])

= Rentabilität: RAROC je Segment
RAROC = (Zinsmarge nach FTP + Provisionen − Betriebskosten − erwarteter Verlust) / ökonomisches Kapital (Euler-Beiträge zum ES).
#figure(image("figures/raroc_by_segment.png", width: 100%), caption: [RAROC je Segment gegen die Hurdle Rate])

#table(columns: 5, align: (left,right, right, right, right), stroke: none, inset: (x: 5pt, y: 3pt),
  fill: (_, y) => if y == 0 { rgb("#ecebe7") },
  table.hline(y: 1, stroke: 0.5pt),
  [*Segment*], [*EAD*], [*ÖK*], [*RAROC*], [*EVA*], [real\_estate\_2], [514,6], [6,8], [49,9 %], [2,71], [real\_estate\_3], [383,8], [11,6], [38,7 %], [3,34], [real\_estate\_1], [318,1], [1,8], [38,3 %], [0,50], [real\_estate\_4], [220,5], [11,4], [32,1 %], [2,51], [trade\_2], [650,5], [12,8], [22,4 %], [1,59], [trade\_3], [511,8], [21,2], [19,3 %], [1,98], [trade\_1], [426,2], [3,5], [14,4 %], [0,15], [manufacturing\_3], [711,1], [47,3], [13,0 %], [1,41], [manufacturing\_2], [903,9], [32,3], [13,0 %], [0,95], [manufacturing\_1], [625,2], [11,4], [7,9 %], [−0,24], [manufacturing\_4], [327,2], [36,4], [6,4 %], [−1,31], [trade\_4], [213,4], [18,1], [5,3 %], [−0,85])

Beträge in Mio. EUR.

= Liquidität und Zinsänderungsrisiko
LCR 239,3 % (HQLA 1.688,3 Mio. EUR, Nettoabflüsse 705,5 Mio. EUR), NSFR 119,1 %. Das Überlebensmodell für Einlagen ergibt einen Bodensatz von 72,6 %; die kumulierte Liquiditätslücke nach einem Jahr ist verhaltensbasiert 295 Mio. EUR statt vertraglich −3.368 Mio. EUR.
#figure(image("figures/irrbb_scenarios.png", width: 100%), caption: [ΔEVE und ΔNII unter den sechs aufsichtlichen Zinsschocks (gestrichelt: Schwellen)])

#table(columns: 5, align: (left,right, right, right, right), stroke: none, inset: (x: 5pt, y: 3pt),
  fill: (_, y) => if y == 0 { rgb("#ecebe7") },
  table.hline(y: 1, stroke: 0.5pt),
  [*Szenario*], [*ΔEVE Mio. EUR*], [*ΔEVE / KK*], [*ΔNII Mio. EUR*], [*ΔNII / KK*], [Parallel aufwärts], [124,5], [17,4 %], [13,1], [1,8 %], [Parallel abwärts], [−152,5], [−21,3 %], [−13,1], [−1,8 %], [Versteilerung], [56,9], [8,0 %], [−4,4], [−0,6 %], [Verflachung], [−37,8], [−5,3 %], [6,6], [0,9 %], [Kurzfristig aufwärts], [3,5], [0,5 %], [10,1], [1,4 %], [Kurzfristig abwärts], [−3,8], [−0,5 %], [−10,1], [−1,4 %])

= Limite und Maßnahmen
#table(columns: 4, align: (left,right, right, right), stroke: none, inset: (x: 5pt, y: 3pt),
  fill: (_, y) => if y == 0 { rgb("#ecebe7") },
  table.hline(y: 1, stroke: 0.5pt),
  [*Limit*], [*Heute*], [*Nach Reallokation*], [*Nach allen Maßnahmen*], [Kreditrisikokapital], [66,8 % grün], [66,8 % grün], [66,8 % grün], [Zinsrisiko ΔEVE], [142,2 % rot], [142,2 % rot], [90,0 % gelb], [Risikodeckungsmasse gesamt], [64,2 % grün], [64,2 % grün], [54,4 % grün], [Zinsergebnis-Schock], [36,5 % grün], [36,5 % grün], [8,5 % grün], [Sektorkonzentration], [98,3 % gelb], [91,6 % gelb], [91,6 % gelb], [Einzelkredit], [111,9 % rot], [110,5 % rot], [100,0 % gelb], [LCR (interne Untergrenze)], [54,3 % grün], [68,0 % grün], [68,0 % grün])

Die Optimierung (cvxpy, EVA-Maximierung unter Kapitalbudget, Sektorgrenzen, Wachstumsgrenzen, LCR- und NSFR-Untergrenzen) hebt den RAROC von 15,9 % auf 17,0 %. Bindend: capital\_budget, growth\_lower, growth\_upper, sector\_share\_real\_estate. Ausbau vor allem in manufacturing\_2, trade\_2, trade\_3, Abbau in manufacturing\_3, manufacturing\_4, real\_estate\_1.

= PD-Kalibrierung für Low-Default-Segmente
Hierarchisches Bayes-Modell (NumPyro) mit Sektorprior und Konjunktureffekt. Auf den 4 Segmenten mit höchstens drei Ausfällen beträgt der mittlere absolute Fehler gegenüber der wahren PD 7,1 bp (Bayes) gegenüber 15,1 bp (naiv) und 13,7 bp (gepoolt). Konformale Intervalle (MAPIE) erreichen 88,9 % Abdeckung bei nominal 90 %.

= Steuerungsagent (lokales LLM)
Der Agent (qwen3:14b) beantwortet Steuerungsfragen nur über fünf freigegebene Werkzeuge und zitiert Regeltexte aus einem lokalen Index. Jede Zahl und jedes Zitat wird vor der Freigabe deterministisch geprüft. Evaluation mit 60 Fragen: 100 % der beantwortbaren und 92 % der mehrstufigen Fragen korrekt, 100 % der themenfremden abgelehnt, 100 % der manipulativen sicher behandelt; 211 freigegebene Zahlen, alle verifiziert. Prompt und Werkzeuge wurden an diesem Fragensatz abgestimmt. Auf 20 vorab festgelegten neuen Fragen: 71 % korrekt, 100 % sicher.

= Betrieb, Monitoring und Hypothesen
Jeder Pipeline-Lauf (Dagster-Job mit Asset-Checks) protokolliert je Modul einen MLflow-Lauf mit Parametern, Metriken, Git-, Daten- und Ergebnis-Hash. Der Drift-Monitor (Evidently) vergleicht Neugeschäfts-Batches mit dem Bestand; ein Immobilienschock wird von der vorab festgelegten Datensatz-Regel nur in 4 von 60 Batches erkannt, von einer Einzelspalten-Regel beim starken Schock in 100 %, ohne Fehlalarme (0 %).
#figure(image("figures/drift_detection.png", width: 100%), caption: [Erkennungsrate nach Schockstärke])

#table(columns: 4, align: (left, left, left, left), stroke: none, inset: (x: 5pt, y: 3pt),
  fill: (_, y) => if y == 0 { rgb("#ecebe7") },
  table.hline(y: 1, stroke: 0.5pt),
  [*ID*], [*Hypothese*], [*Art*], [*Urteil*], [H1], [Granularity adjustment matters above a name HHI of 0.003], [vorab], [gestützt], [H2], [Bayesian shrinkage beats the naive PD on low-default segments], [vorab], [gestützt], [H3], [The drift monitor flags a sector shock and stays quiet without one], [vorab], [nicht gestützt], [H4], [The steering agent generalises to questions it was not tuned on], [vorab], [nicht gestützt], [P1], [Bayesian posterior mean beats naive and pooled PDs on the module-2 history], [nachträglich], [gestützt], [P2], [Sector diversification puts the multi-factor UL below the ASRF UL], [nachträglich], [gestützt], [P3], [Tail dependence (t copula) at least doubles ES], [nachträglich], [gestützt], [P4], [Conformal intervals are close to nominal coverage], [nachträglich], [gestützt], [P5], [Behavioural deposit modelling closes the one-year funding gap], [nachträglich], [gestützt], [P6], [Re-allocation adds at least half a point of RAROC at unchanged capital], [nachträglich], [gestützt])

Hypothesen in englischer Originalfassung; Details in docs/hypotheses.md.

#v(1fr)
#text(8pt, fill: rgb("#52514e"))[Erzeugt mit task report aus results/. Methodik und Annahmen: docs/decisions.md, docs/data.md, docs/versions.md, docs/limits.md.]
