# Data

All generator parameters live in `config/bank.yaml`; this file explains them. Generation is
`task data:synth` (`src/kapitalkompass/data/synth.py`), seeded with `seed` from the config.
Content hashes of every generated table are locked in `data/synthetic/HASHES` (committed);
`tests/test_synth.py::test_matches_locked_hashes` fails if output drifts by one bit.

## Synthetic loan book (`data/synthetic/loan_book.parquet`)

One row per obligor (one facility per obligor). Draw order is fixed, all draws from one
`numpy.random.default_rng(seed)` (PCG64).

| Column | Generator assumption |
|---|---|
| `sector` | Categorical: manufacturing 40%, trade 35%, real estate 25%. |
| `grade` | Rating grades 1-4 with shares 25/35/28/12%, independent of sector. |
| `segment` | `sector_grade`, 12 segments. These are the steering units for RAROC and limits. |
| `pd` | Through-the-cycle one-year PD per grade: 0.15%, 0.50%, 1.50%, 5.00%. Grade-level, no obligor noise (the Bayesian module estimates these back from simulated defaults). |
| `ead_eur` | Lognormal, median 50k EUR, sigma 1.2 (mean about 103k). 25 large names in manufacturing and real estate replaced by Pareto draws (min 10m, alpha 1.6, capped at 120m) to create name concentration. |
| `lgd` | Beta with sector mean (manufacturing 40%, trade 45%, real estate 25%, reflecting collateral) and concentration a+b=10, clipped to [5%, 95%]. Obligor-level LGD is deterministic in the loss model (no LGD-PD correlation). |
| `maturity_years` | Gamma(shape 2) around the sector mean (4, 3, 7 years), clipped to [0.25, 15]. |
| `fixed_rate` | 60% fixed-rate, rest floating (reprice within 3 months). Used for IRRBB repricing. |
| `client_margin` | Grade margin over the base rate (1.00/1.50/2.30/3.50%) plus N(0, 0.25%) noise. |
| `fee_rate` | Exponential, mean 0.20% of EAD, capped at 5%. |
| `opex_rate` | Sector cost-to-serve: 0.60/0.70/0.40% of EAD. |
| `is_large` | Flag for the 25 large names. |

Result: about 5.8 bn EUR of loans across 50,000 obligors (the exact figure is in the generated
data, not typed here).

## Synthetic balance sheet (`data/synthetic/balance_sheet.parquet`)

Total assets are fixed by loans = 65% of total assets. Every other line is a share of total
assets (see config). Amounts are integer EUR. `wholesale_unsecured_gt1y` is the residual,
so assets = liabilities + equity holds exactly (pandera check `identity_holds`). CET1 equity
is 8% of total assets. `maturity_years` is the contractual repricing maturity used by the
ALM module (0 = overnight or non-maturity; behavioural assumptions for non-maturity deposits
come from the survival model in module 3, not from this column).

HQLA classes follow the LCR Delegated Regulation (EU) 2015/61 levels (L1 cash and central-bank
reserves, L1 government bonds, L2A covered bonds, L2B corporate bonds). Haircuts and caps are
applied in module 3 and recorded in `docs/versions.md`.

## External data

| Data | Source | Cache | Licence |
|---|---|---|---|
| Euro-area AAA government spot curve (Svensson, continuous compounding), tenors 3M-30Y, daily since 2020-01-01 up to the reporting date | ECB Data Portal, dataset `YC`, series `B.U2.EUR.4F.G_N_A.SV_C_YM.SR_*` | `data/external/ecb_yc_spot.parquet` (committed) | ECB statistics, reuse permitted with source attribution |

The pipeline reads only the cache; `task data:rates` refreshes it (network). Rates are
stored as decimals.
