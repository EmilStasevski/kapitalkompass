# Kapitalkompass

[Deutsche Fassung](README.de.md)

A Gesamtbanksteuerung (bank-wide steering) cockpit for a stylised bank: where is capital consumed, what does it earn, can it be funded in liquidity, and which management action improves the picture? It covers credit portfolio risk with concentration, Bayesian calibration for low-default portfolios, ALM and liquidity (LCR/NSFR, IRRBB/NII), RAROC and capital allocation, constrained optimisation with a limit system, and a local LLM agent that answers steering questions through audited tool calls. Everything runs locally on one 16 GB GPU. Every number in this README is generated from `results/` by `task report`.

> **Not investment advice, not a regulatory submission.** This is a risk-steering prototype on synthetic and public data. Regulatory formulas were checked against the primary text (`docs/versions.md`).

## Headline results

<!-- BEGIN GENERATED: scripts/report_readme.py -->

| Module | Headline (all numbers generated from `results/`) |
|---|---|
| 1, credit portfolio model | TBD obligors, 3 sectors x 4 rating grades: 99.9% credit VaR TBD m EUR, ES TBD m EUR; Monte Carlo vs analytic ASRF in the granular limit: TBD% deviation; GPU vs CPU speed-up TBD x |
| 2, Bayesian PD calibration | Low-default segments: posterior PD vs pooled vs naive, 95% intervals; coverage of conformal intervals TBD% |
| 3, ALM and liquidity | LCR TBD%, NSFR TBD%; NII change under 6 shock scenarios: worst TBD m EUR; behavioural deposit model vs contractual: TBD |
| 4, RAROC and capital allocation | Bank RAROC TBD% vs hurdle TBD%; Euler allocation sums to total: yes; TBD of 12 segments below hurdle |
| 5, optimisation and limits | Re-allocation raises RAROC from TBD% to TBD% at unchanged capital; limit system: TBD green / TBD amber / TBD red |
| 6, LLM steering agent | TBD questions: TBD% tool-call correct, TBD% numbers verified, TBD% citations resolve to retrieved text, TBD% out-of-scope refused |
| 7, MLOps | MLflow runs TBD; drift monitor flags the injected shift in TBD of TBD cases |

<!-- END GENERATED -->

## Why this project

Risk measurement alone does not run a bank. Gesamtbanksteuerung links measured risk to capital, earnings and liquidity and ends in decisions. Kapitalkompass builds that chain end to end:

```
loan book + balance sheet --> credit portfolio loss distribution --> economic capital
                                                                           |
deposits + market rates --> ALM / LCR / NSFR / NII --> FTP --> RAROC per segment --> limits --> optimiser --> management report (LLM agent)
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
| 7, MLOps | Are runs reproducible and is drift detected? | `reports/mlops.md` |

## How it is built

- **Data:** seeded synthetic loan book and balance sheet (generator documented in `docs/data.md`), ECB/Bundesbank rate curves, public sector insolvency statistics as PD priors, EBA/BaFin texts for the RAG corpus. Licences: `docs/data.md`.
- **Discipline:** analytic cross-checks as tests (Monte Carlo vs ASRF, Euler additivity, LCR/NSFR identities), time-based splits for the Bayesian module, pandera data contracts, deterministic verification of every LLM output.
- **Stack:** Python 3.12 with uv; Polars + pandera; PyTorch CUDA; NumPyro; lifelines; MAPIE; cvxpy; MLflow; Evidently; Dagster; LanceDB + bge-m3; local Qwen3-14B or gpt-oss-20b (quantised, fits 16 GB); MCP server (Python SDK); Typst for PDF reports; pytest, ruff, mypy.

## Quick start

```bash
uv sync
task data:synth     # generate the synthetic bank (seeded)
task test           # unit + analytic cross-check tests
task pipeline       # Dagster job: modules 1-5, writes results/
task report         # rebuild reports/ and the tables above
task agent:eval     # module 6 evaluation (needs the local LLM server)
```

See [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md) for the one-week plan.
