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
