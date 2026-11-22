# Decisions

Readings chosen where the instructions leave room. Newest last.

| # | Date | Decision | Reason |
|---|---|---|---|
| 1 | 2026-11-22 | `README_dr.md` renamed to `README.md`. | The build prompt refers to `README.md`; the `_dr` file is the draft of it. |
| 2 | 2026-11-22 | `task` (go-task) is installed as a dev dependency (`go-task-bin`), run as `uv run task …`. | Not on the machine; keeps the toolchain inside the uv lockfile. |
| 3 | 2026-11-22 | Segments = sector x rating grade (3 x 4 = 12). | Matches "3 sectors, 4 rating grades, 12 segments" exactly. |
| 4 | 2026-11-22 | One facility per obligor. | Keeps name concentration and obligor counts identical; multi-facility adds nothing to the steering logic. |
| 5 | 2026-11-22 | Rate curve: ECB AAA Svensson spot curve only; no separate Bundesbank series. | The ECB curve gives every tenor needed for FTP and IRRBB from one consistent fitted model; Bundesbank term structure would duplicate it. |
| 6 | 2026-11-22 | Rate cache committed to git; pipeline never touches the network. | "`task pipeline` reproduces `results/` bit-for-bit" is impossible if the curve is re-downloaded. |
| 7 | 2026-11-22 | Reproducibility is checked on content hashes (CSV serialisation), not parquet bytes. | Parquet files embed writer version metadata; content hash is the meaningful invariant. |
| 8 | 2026-11-22 | Reporting date 2026-11-20; the ECB curve for that day was not yet published, so `curve_at` uses the last available date on or before it (2026-11-19). | Standard last-available-observation convention. |
| 9 | 2026-11-22 | `README.de.md` (linked from README) is generated together with the README on day 7. | It must carry generated numbers too. |
| 10 | 2026-11-23 | Regulatory formulas are checked against the BCBS texts (BIS Basel Framework chapter PDFs, BCBS 368) instead of CRR/EU delegated acts. | EUR-Lex returned empty HTTP 202 responses to the automated fetcher. The EU texts transpose these standards; known or possible EU deviations are flagged per row in `docs/versions.md`. |
| 11 | 2026-11-23 | Credit model: sector factors Z_s = √w·G + √(1−w)·S_s with w = 0.55 and sector loadings ρ = 0.16/0.14/0.20 (`config/model.yaml`). | Gives an inter-sector factor correlation of 0.55 and within-sector asset correlations close to the Basel range at low PD. |
| 12 | 2026-11-23 | Economic capital = ES₉₉.₉ − EL; segment capital = Euler ES contribution − segment EL. | ES contributions E[L_g \| L ≥ VaR] are additive by construction and far less noisy than VaR contributions (also reported). |
| 13 | 2026-11-23 | The ASRF cross-check compares simulated VaR − EL with the analytic K (no maturity term). A one-factor run on the real book isolates name concentration; the main multi-factor run sits below ASRF because of sector diversification. | ASRF is exactly the one-factor model in the infinitely granular limit. |
| 14 | 2026-11-23 | Pipeline: 2M Gaussian scenarios; t copula (ν = 6) and one-factor runs with 1M scenarios each. GPU runs are deterministic per device and driver; CPU runs (`--device cpu`) are bit-for-bit. | Keeps `task pipeline` under a few minutes on the GPU. |
| 15 | 2026-11-23 | The benchmark is a separate `task bench` writing `results/credit/benchmark.json`, not part of `task pipeline`. | Timings and energy depend on hardware and load, so they cannot be reproduced bit-for-bit. |
| 16 | 2026-11-23 | Sector prior centres for the Bayesian PD model (0.8%, 1.0%, 0.5% average-grade PD, logit sd 0.75) are stated assumptions, not public statistics. | Destatis publishes total corporate insolvency counts (2025: 24,064, table lrins01) but no sector insolvency rates were retrievable (GENESIS 52411 is a dynamic page). The wide prior sd keeps the assumption weak; replace it with GENESIS 52411 rates when available. |
| 17 | 2026-11-23 | The Bayesian model has a common year effect, and the TTC PD integrates it out (Gauss-Hermite). Conservative PD = upper bound of the 90% credible interval. "Pooled" = grade-level pooled default rate across sectors. | Annual default rates are cycle-driven; ignoring the shared year effect would understate uncertainty. The upper credible bound gives a margin-of-conservatism-style PD. |
| 18 | 2026-11-23 | Conformal counterpart: MAPIE CV+ intervals for annual segment default rates from a ridge model on grade, sector and the macro index, fitted on arcsin(√rate) and back-transformed. | Conformal classification sets are uninformative for rare binary defaults. The variance-stabilising transform is monotone, so coverage is preserved and the intervals scale with grade. |
| 19 | 2026-11-23 | Truth recovery is checked by refitting on fresh simulated histories (40 replications in the pipeline, 16 in the unit test) at 50/80/90% levels. | Segments in one history share the cycle draw, so coverage from a single fit is strongly clustered; 96/96 hits in a pilot with 8 replications was not evidence of miscalibration. |
