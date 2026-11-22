# Kapitalkompass: one-week build instructions

Goal: a second portfolio project that shows the part your first one (Risikopegel) does not: steering. Risikopegel measures risk, Kapitalkompass turns it into capital allocation, liquidity, limits and decisions. Target job: Werkstudent financial data science with Finance, Risikomanagement, Banksteuerung, Risikotragfähigkeit, Gesamtbanksteuerung.

## What is new compared with Risikopegel

| Area | Risikopegel (already done) | Kapitalkompass (new) |
|---|---|---|
| Credit | PD scorecard vs GBM, ECL | Portfolio loss distribution, concentration, Euler capital allocation, Bayesian low-default PD |
| Market / ALM | VaR/ES backtests, IRRBB outlier flag | Deposit behaviour (survival analysis), LCR/NSFR, NII under 6 shocks, FTP |
| Steering | RTF stylised bank | RAROC per segment, limit system, constrained optimiser |
| LLM | Verified German explanations, ECB tone | Tool-calling agent + RAG with citations, MCP server |
| Engineering | dbt, DuckDB, FastAPI, Docker, Streamlit | Polars, pandera, CUDA, MLflow, Evidently, Dagster, Typst |
| Statistics | GARCH, GBM, foundation models | Bayesian (NumPyro), conformal prediction (MAPIE), copulas, convex optimisation |

## Ground rules for the automated run

1. Everything is seeded; `task pipeline` reproduces `results/` bit-for-bit on CPU and within tolerance on GPU.
2. No number in README or `reports/` is typed by hand. Generated from `results/`.
3. Every module ships one analytic cross-check test. If it fails the module is not done.
4. Regulatory formulas go into `docs/versions.md` with the source passage checked, not recalled. Use WebFetch on the primary text (CRR, EBA IRRBB guidelines, BCBS 368) and record the paragraph number.
5. LLM output is never trusted: numbers are matched against `results/`, citations must resolve to retrieved chunks, tool calls go through an allowlist with Pydantic schemas.
6. Stop rule: modules 1-5 are the MVP. If behind schedule, cut module 7 first, then the MCP server in module 6.

## Day plan

### Day 1: foundation and synthetic bank
- uv project, ruff, mypy, pytest, Taskfile, pre-commit.
- Generator (Polars): 50k obligors, 3 sectors, 4 rating grades, 12 segments, exposure, LGD, maturity; balance sheet with loans, bonds, HQLA, retail and corporate deposits, wholesale funding, equity.
- pandera contracts for every table. Rate curves from ECB/Bundesbank APIs, cached as parquet.
- Done when: `task data:synth` is deterministic, schema tests pass, `docs/data.md` documents every generator assumption.

### Day 2: module 1, credit portfolio model (GPU)
- One-factor and sector multi-factor Gaussian copula; optional t-copula for tail dependence. Loss simulation in PyTorch CUDA, 1-10 million scenarios, chunked for 16 GB.
- Outputs: loss distribution, expected loss, 99.9% credit VaR and ES, HHI and name concentration, Euler contributions per segment.
- Cross-check tests: for a granular homogeneous portfolio, simulated 99.9% loss quantile matches the analytic ASRF formula `K = LGD * [N((N^-1(PD) + sqrt(rho) * N^-1(0.999)) / sqrt(1 - rho)) - PD]` within tolerance (verify against CRR text); Euler contributions sum to total ES.
- Benchmark table: numpy CPU vs numba vs CUDA, time and energy (nvidia-smi sampling).

### Day 3: module 2, Bayesian PD calibration and module 3 start
- NumPyro hierarchical model: segment PDs shrink to a sector prior, prior informed by public insolvency statistics. Compare with naive and pooled estimates on segments with 0-3 defaults; posterior predictive check; conservative PD at an upper credible bound (margin-of-conservatism idea).
- MAPIE conformal intervals on a simple PD model as the frequentist counterpart; report coverage.
- Done when: simulated truth recovery test passes (known PDs in, interval covers truth at about the nominal rate).

### Day 4: module 3, ALM, liquidity, IRRBB
- lifelines survival model for deposit run-off and core share from a simulated behaviour set; compare behavioural vs contractual cash-flow ladder.
- LCR (HQLA / net 30-day outflows, minimum 100%) and NSFR (ASF / RSF, minimum 100%) with the factor tables entered from the primary text.
- NII and EVE under the six supervisory shocks (parallel up/down, steepener, flattener, short up, short down); outlier test thresholds taken from the EBA guideline text.
- FTP curve: base rate plus liquidity premium by tenor; output FTP charge per segment.
- Tests: balance sheet identity, LCR/NSFR hand-computed toy case, zero-shock gives zero delta.

### Day 5: modules 4 and 5, RAROC, optimisation, limits
- RAROC per segment = (net interest margin after FTP + fees - opex - expected loss) / allocated economic capital (Euler from module 1). Hurdle rate as input.
- cvxpy: maximise portfolio RAROC or risk-adjusted profit subject to capital budget, sector concentration limits, LCR floor and growth bounds. Show which constraint binds and its shadow price.
- Limit system: traffic lights (green / amber / red) on utilisation of risk cover, concentration, LCR, NII shock, with a documented escalation rule.
- Done when: allocation additivity test, optimiser never violates constraints, a tiny hand-solved LP matches.

### Day 6: module 6, LLM steering agent
- Model: Qwen3-14B Q4 or gpt-oss-20b (quantised) via Ollama or llama.cpp, plus bge-m3 embeddings; both fit in 16 GB if the embedder is run on demand. Check current model tags before pulling.
- RAG corpus: EBA IRRBB guidelines, BCBS LCR/NSFR texts, MaRisk (BaFin), CRR extracts. LanceDB index, chunk ids stable.
- Tools (Pydantic schemas, allowlist): `get_raroc(segment)`, `get_limits()`, `run_shock(name)`, `run_optimiser(constraints)`, `search_rules(query)`. JSON-schema constrained decoding.
- Output: German Vorstandsvorlage of one page: situation, drivers, recommended action, rule citations.
- Deterministic verification: every number matches a tool result; every citation id exists in retrieved chunks; out-of-allowlist or out-of-scope questions must be refused.
- Eval set: 60 generated questions (24 answerable, 12 multi-step, 12 out-of-scope, 12 adversarial, e.g. asks for a made-up figure). Report rates. Optional: expose the same tools as an MCP server.

### Day 7: module 7, MLOps, report, polish
- MLflow logs every run (params, metrics, git hash, data hash). Evidently drift report with an injected sector shock; check it is flagged.
- Dagster job for the pipeline with asset checks.
- Typst or Quarto PDF "Gesamtbank-Steuerungsbericht" generated from `results/`.
- `task report` rebuilds README tables, `reports/`, PDF. CI runs lint, mypy, tests (CPU subset).
- Write `docs/hypotheses.md` with pre-registered hypotheses and verdicts (e.g. "granularity adjustment matters above HHI X", "Bayesian shrinkage beats naive PD on 0-default segments").

## Prompt to start the autonomous run (Claude Code)

```
Read README.md and BUILD_INSTRUCTIONS.md in this repo. Build the project day by day
exactly as specified. Rules: seed everything; no hand-typed numbers in README or
reports; each module needs its analytic cross-check test passing before moving on;
verify every regulatory formula against the primary text and record it in
docs/versions.md; verify all LLM output deterministically as described. Work
without asking questions: choose the most reasonable reading, record it in
docs/decisions.md. After each day, run task test and task lint, commit, and write
a 5-line status to STATUS.md. If a module cannot be finished, cut it per the stop
rule and note it in STATUS.md. Finish by running task pipeline and task report.
```

## GPU notes (16 GB VRAM)

- Monte Carlo: chunk scenarios so one chunk uses under 6 GB; keep the LLM server stopped during module 1 benchmarks, otherwise timings are unfair.
- LLM: one 14B-20B quantised model at a time, context 8k. Run the embedder on CPU if memory is tight.
- Use `torch` with CUDA here (Risikopegel used CPU-only torch); keep separate environments if needed.

## How to present it

Three-sentence pitch (German): "Risikopegel misst Risiken, Kapitalkompass steuert damit die Bank: Aus einem Kreditportfoliomodell leite ich ökonomisches Kapital ab, verteile es per Euler-Allokation auf Segmente und messe RAROC nach Refinanzierungskosten. Dazu kommen Liquidität (LCR/NSFR), Zinsänderungsrisiko und ein Limitsystem mit Optimierer, das Maßnahmen vorschlägt. Ein lokaler LLM-Agent berichtet darüber, und jede Zahl und jedes Zitat wird deterministisch geprüft."

CV lines:
- Built a Gesamtbanksteuerung prototype: GPU Monte Carlo credit portfolio model validated against the Basel ASRF formula, Euler capital allocation, RAROC and constrained re-allocation (cvxpy).
- Modelled ALM and liquidity: behavioural deposits (survival analysis), LCR/NSFR, NII/EVE under six supervisory rate shocks, FTP.
- Bayesian PD calibration for low-default portfolios (NumPyro) with conformal intervals.
- Local LLM agent with audited tool calls and RAG over EBA/BaFin texts; 100% of reported numbers verified against results.

## Optional extras if time remains

- Power BI or Excel pack: star-schema parquet/CSV export and an `xlsxwriter` workbook with live formulas for the RAROC table, since many banking teams work in Excel and Power BI.
- Reverse stress test on the optimiser (find the shock that makes the capital constraint bind).
- Link to Risikopegel: import its PD and macro-scenario outputs as inputs to module 1 and cite it in the README.
