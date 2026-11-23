# Kapitalkompass

[Deutsche Fassung](README.de.md)

**A Gesamtbanksteuerung (bank-wide steering) cockpit, built end to end on one GPU.** Kapitalkompass takes a bank from risk measurement to decisions: a GPU credit portfolio model yields economic capital, Euler allocation distributes it to segments, RAROC after funding cost shows who earns their capital, liquidity (LCR/NSFR) and interest-rate risk (IRRBB) set the boundaries, and a constrained optimiser with a limit system proposes management actions. A local LLM agent writes the board memo, and every number and citation it releases is checked deterministically.

> Prototype on synthetic and public data; not investment advice and not a regulatory submission.

<!-- BEGIN GENERATED: kapitalkompass.report -->

## Highlights

- **Bank-scale GPU Monte Carlo:** 2,000,000 correlated default scenarios over 50,000 obligors on an NVIDIA GeForce RTX 5060 Ti, chunked for 16 GB VRAM; 177x faster than numpy, 26x faster than parallel numba, and 74x less energy per scenario.
- **Validated against Basel, not just simulated:** the one-factor simulation matches the analytic ASRF/IRB formula within 2.2%, and the gap is the measured granularity add-on; segment Euler ES contributions add up exactly to total ES.
- **Tail risk made visible:** a t-copula (ν = 6) shows 3.0x the Gaussian ES, the model-risk number a steering committee needs to see.
- **Reliable PDs from few defaults:** hierarchical Bayesian estimation (NumPyro) cuts the error in low-default segments 2.1x versus the naive estimate, which returns a PD of zero for segments without defaults.
- **From measurement to decisions:** the optimiser (cvxpy) lifts bank RAROC from 15.9% to 17.0% at unchanged capital and reports shadow prices of the binding constraints; together with a 599m EUR swap hedge, red limits fall from 2 to 0.
- **An LLM you can audit:** all 211 released numbers and 93 citations are deterministically verified against tool output and retrieved regulatory text; 100% of adversarial prompts handled safely; runs fully local (qwen3:14b).
- **Regulation from the primary text:** formulas, shocks and thresholds from the CRR, CRD, EBA RTS 2024/856, BCBS 368 and the Basel LCR/NSFR standards, with paragraph references in `docs/versions.md`.
- **Reproducible by construction:** everything is seeded; a rerun reproduces 241 of 241 metrics exactly; no number in the README or reports is typed by hand, and a test fails as soon as this file drifts from `results/`.

## Headline results by module

| Module | Headline (all numbers generated from `results/`) |
|---|---|
| 1, credit portfolio model | 50,000 obligors, 3 sectors x 4 rating grades: 99.9% credit VaR 203.9m EUR, ES 242.3m EUR; one-factor Monte Carlo vs analytic ASRF: 2.2% deviation (granularity add-on); GPU vs CPU (numpy) speed-up 177x |
| 2, Bayesian PD calibration | 4 low-default segments: mean absolute error Bayes 7.1 bp vs pooled 13.7 bp vs naive 15.1 bp; conformal coverage 88.9% (nominal 90%) |
| 3, ALM and liquidity | LCR 239.3%, NSFR 119.1%; worst NII change over 6 shocks −13.1m EUR, worst ΔEVE −21.3% of Tier 1 (outlier: yes); cumulative 1-year gap behavioural 295m EUR vs contractual −3,368m EUR |
| 4, RAROC and capital allocation | Bank RAROC 15.9% vs hurdle 10%; Euler allocation sums to total: yes; 3 of 12 segments below hurdle |
| 5, optimisation and limits | Re-allocation raises RAROC from 15.9% to 17.0% at unchanged capital (re-simulated); limit system 4 green / 1 amber / 2 red before, 4 green / 3 amber / 0 red after actions (599m EUR swap hedge) |
| 6, LLM steering agent | 60 questions: answerable 100% correct, multi-step 92%, out-of-scope 100% refused, adversarial 100% safe; all 211 released numbers verified (first attempt 92%) |
| 7, MLOps | Dagster job with asset checks; 7 MLflow runs per pipeline run (params, metrics, git, data and results hash); a rerun reproduces 241 of 241 metrics exactly; the Evidently drift monitor catches the strong sector shock in 100% of batches with 0% false alarms |

<!-- END GENERATED -->

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
