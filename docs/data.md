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

## Synthetic default history (`data/synthetic/default_history.parquet`)

Input for the Bayesian PD module. One row per segment and year, 2014-2025; years up to 2022
are the training sample, 2023-2025 the held-out test years (time split).

| Column | Generator assumption |
|---|---|
| `obligors` | Rated obligors observed in that segment-year: 110/180/380/280 for grades 1-4, scaled by 1.0/0.8/0.5 for manufacturing/trade/real estate. Small grade-1 and grade-2 cohorts create low-default segments (0-3 defaults over the training years). |
| `defaults` | Binomial(obligors, p_t) with the Vasicek conditional PD p_t = N((N^-1(PD) − √ρ·z_t)/√(1−ρ)), ρ = 0.08, z_t ~ N(0,1) shared by all segments in year t. E[p_t] = PD exactly (test `test_vasicek_year_pd_averages_to_ttc`). |
| `true_pd` | The through-the-cycle PD of the grade from the loan book, the truth the Bayesian model is scored against. |
| `macro` | Observable macro index m_t = −0.8·z_t + 0.6·noise, so bad years show low m_t. Used only by the conformal counterpart model. |

## Regulatory corpus (module 6)

The agent's rule search runs over paragraph-level chunks of ten texts, pinned by sha256 in
`config/corpus.yaml`. The PDFs are third-party texts and are not committed; they live in
`data/external/regs/` (see `url` per document). `data/corpus_chunks.csv` (committed) lists every
chunk id, its citation label, length and a hash of its text, so a test can confirm that
re-chunking reproduces exactly the same ids.

| id | Document | Chunked by |
|---|---|---|
| `crr` | CRR consolidated 26.06.2026, extracts: Art. 92, 153, 395, 412, 413, 428b, 428k–428o, 428r, 428s, 428u, 428ad, 428ag, 428ah | numbered paragraph (`crr.art395.1`) |
| `crd` | CRD consolidated 17.01.2025, extracts: Art. 73, 84, 98 | numbered paragraph (`crd.art98.5`) |
| `rts856` | Delegated Regulation (EU) 2024/856 (IRRBB supervisory outlier tests), Art. 1–5 | numbered paragraph |
| `ebagl` | EBA/GL/2022/14 IRRBB/CSRBB guidelines, section 4 (the guidelines proper) | Randnummer (`ebagl.p115`) |
| `marisk` | BaFin Rundschreiben 05/2023 (BA) MaRisk | module and Textziffer (`marisk.btr3.1.tz4`) |
| `bcbs368` | BCBS 368 IRRBB standard (2016), paras 1–132 and Annexes 1–2 | paragraph / annex (`bcbs368.p88`, `bcbs368.annex2`) |
| `lcr30`, `lcr40`, `nsf30`, `cre31` | Basel Framework chapters (BIS chapter PDFs) | Basel paragraph (`lcr30.40`) |

Text is extracted with poppler's `pdftotext`; page headers, page numbers and EUR-Lex amendment
markers are removed; paragraphs longer than 2,400 characters are split into `#2`, `#3` parts.
Formulas that are images in the PDFs (e.g. CRE31.5, LCR30.38) are missing from the chunk text.
Embeddings: `bge-m3` via Ollama, stored in LanceDB (`data/index/`, not committed, rebuilt by
`task agent:index`). Ollama 0.13.5 returns NaN for a handful of short inputs with bge-m3; those
are re-embedded with trailing words dropped one at a time (deterministic).
