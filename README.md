# Kapitalkompass

[Deutsche Fassung](README.de.md)

A Gesamtbanksteuerung (bank-wide steering) cockpit for a stylised bank: where is capital consumed, what does it earn, can it be funded in liquidity, and which management action improves the picture? It covers credit portfolio risk with concentration, Bayesian calibration for low-default portfolios, ALM and liquidity (LCR/NSFR, IRRBB/NII), RAROC and capital allocation, constrained optimisation with a limit system, and a local LLM agent that answers steering questions through audited tool calls. Everything runs locally on one 16 GB GPU. Every number in this README is generated from `results/` by `task report`.

> **Not investment advice, not a regulatory submission.** This is a risk-steering prototype on synthetic and public data. Regulatory formulas were checked against the primary text (`docs/versions.md`).

## Headline results

<!-- BEGIN GENERATED: kapitalkompass.report -->

| Module | Headline (all numbers generated from `results/`) |
|---|---|
| 1, credit portfolio model | 50,000 obligors, 3 sectors x 4 rating grades: 99.9% credit VaR 203.9m EUR, ES 242.3m EUR; one-factor Monte Carlo vs analytic ASRF: 2.2% deviation (granularity add-on); GPU vs CPU (numpy) speed-up 177x |
| 2, Bayesian PD calibration | 4 low-default segments: mean absolute error Bayes 7.1 bp vs pooled 13.7 bp vs naive 15.1 bp; conformal coverage 88.9% (nominal 90%) |
| 3, ALM and liquidity | LCR 239.3%, NSFR 119.1%; worst NII change over 6 shocks −13.1m EUR, worst ΔEVE −21.3% of Tier 1 (outlier: yes); cumulative 1-year gap behavioural 295m EUR vs contractual −3,368m EUR |
| 4, RAROC and capital allocation | Bank RAROC 15.9% vs hurdle 10%; Euler allocation sums to total: yes; 3 of 12 segments below hurdle |
| 5, optimisation and limits | Re-allocation raises RAROC from 15.9% to 17.0% at unchanged capital (re-simulated); limit system 4 green / 1 amber / 2 red before, 4 green / 3 amber / 0 red after actions (599m EUR swap hedge) |
| 6, LLM steering agent | 60 questions: answerable 100% correct, multi-step 92%, out-of-scope 100% refused, adversarial 100% safe; all 211 released numbers verified (first attempt 92%); held-out (20 new questions): 71% correct, 100% safe |
| 7, MLOps | 7 MLflow runs per pipeline run (params, metrics, git, data and results hash); drift monitor: the pre-registered dataset rule flags 4 of 60 shocked batches, the column-level rule 100% at the strong shock, false alarms 0%; 2 of 4 pre-registered hypotheses supported |

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
| 7, MLOps | Are runs reproducible and is drift detected? | `reports/mlops.md`, `docs/hypotheses.md` |

The German management report `reports/Gesamtbank-Steuerungsbericht.pdf` (Typst) and the agent's
verified board memo `reports/vorstandsvorlage.md` are generated from the same `results/`.

## How it is built

- **Data:** seeded synthetic loan book and balance sheet (generator documented in `docs/data.md`), ECB/Bundesbank rate curves, public sector insolvency statistics as PD priors, EBA/BaFin texts for the RAG corpus. Licences: `docs/data.md`.
- **Discipline:** analytic cross-checks as tests (Monte Carlo vs ASRF, Euler additivity, LCR/NSFR identities), time-based splits for the Bayesian module, pandera data contracts, deterministic verification of every LLM output.
- **Stack:** Python 3.12 with uv; Polars + pandera; PyTorch CUDA; NumPyro; lifelines; MAPIE; cvxpy; MLflow; Evidently; Dagster; LanceDB + bge-m3; local Qwen3-14B via Ollama (quantised, fits 16 GB); MCP server (Python SDK); Typst for PDF reports; pytest, ruff, mypy.

## Quick start

```bash
uv sync
task data:synth     # generate the synthetic bank (seeded)
task test           # unit + analytic cross-check tests
task pipeline       # Dagster job: modules 1-5, drift monitor, hypothesis experiments; MLflow runs
task report         # rebuild reports/, the PDF, docs/hypotheses.md and the tables above
task agent:eval     # module 6 evaluation (needs the local LLM server); agent:heldout for H4
task mlflow:ui      # browse the tracked runs; dagster:ui for the job graph
```

See [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md) for the one-week plan.
