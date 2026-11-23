# Regulatory formulas and their sources

Every regulatory formula or parameter used in code is recorded here with the primary text and
paragraph, read from the source document in this project (not recalled).

**Source access note (2026-11-23).** EUR-Lex (CRR 575/2013, Delegated Regulation 2015/61,
RTS 2024/856) answered automated requests with an empty HTTP 202 page, so the EU texts could
not be read in the automated run. The Basel Committee texts that the EU rules transpose were
read instead (BIS consolidated Basel Framework chapter PDFs and BCBS 368). Where the EU text is
known to differ or could not be confirmed, the row says so. See decision 10 in
`docs/decisions.md`.

**Update (Day 6, 2026-11-23).** Local copies of the consolidated CRR (version of 26.06.2026), the
consolidated CRD (17.01.2025) and Delegated Regulation (EU) 2024/856 (OJ L, 24.4.2024), downloaded
from EUR-Lex in an earlier session on this machine, were found and added to the RAG corpus
(`config/corpus.yaml`, sha256-pinned). The EU rows below marked "Day 6" were checked against
them. Delegated Regulation 2015/61 (EU LCR) is still not available, so the EU LCR deviations
remain unverified.

| Module | Formula / parameter | Value used | Primary source | Paragraph | Checked on | Code |
|---|---|---|---|---|---|---|
| 1 | Corporate asset correlation R(PD) | 0.12·(1−e^(−50PD))/(1−e^(−50)) + 0.24·[1−(1−e^(−50PD))/(1−e^(−50))] | BCBS Basel Framework CRE31, version effective 01 Jan 2023 (chapter PDF) | CRE31.5 | 2026-11-23 | `credit/asrf.py::basel_corporate_rho` |
| 1 | Maturity adjustment b | [0.11852 − 0.05478·ln(PD)]² | same | CRE31.5 | 2026-11-23 | `maturity_adjustment_b` |
| 1 | Capital requirement K | [LGD·N(G(PD)/√(1−R) + √(R/(1−R))·G(0.999)) − PD·LGD]·(1+(M−2.5)b)/(1−1.5b) | same | CRE31.5, CRE31.6 | 2026-11-23 | `irb_k`, `asrf_k` (without maturity term) |
| 1 | RWA | K·12.5·EAD (no 1.06 scaling in the Basel III text; CRR wording not verified) | same | CRE31.5 | 2026-11-23 | reported only |
| 1 | N, G, PD/LGD as decimals | definitions | same | CRE31.2 | 2026-11-23 | — |
| 1 | Effective maturity M | Defined in CRE32.43–32.54; this project clips contractual maturity to [1, 5] years as a proxy (floor/cap not verified verbatim) | CRE31.6 (reference) | CRE31.6 | 2026-11-23 | `credit/run.py` |
| 1 | Corporate PD floor | 0.05% (all synthetic PDs ≥ 0.15%, so it never binds) | BCBS CRE32 | CRE32.4 | 2026-11-23 | not needed |
| 3 | Six IRRBB shock scenarios | parallel up/down, steepener, flattener, short up, short down | BCBS 368 (April 2016) | Annex 2, p. 44 | 2026-11-23 | module 3 |
| 3 | EUR shock sizes | parallel 200 bp, short 250 bp, long 100 bp | BCBS 368 | Annex 2, Table 1 | 2026-11-23 | module 3 |
| 3 | Shock shapes | ΔR_short(t) = ±R̄_short·e^(−t/x), ΔR_long(t) = ±R̄_long·(1−e^(−t/x)), x = 4 | BCBS 368 | Annex 2 (i)–(iii), footnote 43 | 2026-11-23 | module 3 |
| 3 | Rotation shocks | steepener = −0.65·\|ΔR_short\| + 0.9·\|ΔR_long\|; flattener = +0.8·\|ΔR_short\| − 0.6·\|ΔR_long\| | BCBS 368 | Annex 2 (iv) | 2026-11-23 | module 3 |
| 3 | Post-shock floor | BCBS: national discretion, floors not greater than zero. **EU (used):** −150 bp at immediate maturity, +3 bp per year, 0% from 50 years; observed rates below the floor are kept (Day 6). Non-binding at current EUR rates, results unchanged | BCBS 368; Delegated Regulation (EU) 2024/856 | Annex 2 footnote 46; Art. 3(7) | 2026-11-23 | `alm/curves.py::eu_floor` |
| 3 | EVE outlier test | max ΔEVE over the six scenarios compared with 15% of Tier 1 | BCBS 368 | para 88 (Principle 12) | 2026-11-23 | module 3 |
| 3 | NII / additional outlier tests | at least as stringent as 15% of Tier 1 | BCBS 368 | para 89 | 2026-11-23 | module 3 |
| 3 | EU shock sizes for EUR (Day 6) | parallel 200 bp, short 250 bp, long 100 bp: identical to BCBS 368 | Delegated Regulation (EU) 2024/856 | Annex, Part A | 2026-11-23 | `config/alm.yaml` irrbb.shocks_bp |
| 3 | EU shock shapes (Day 6) | parallel, short (e^(−t/4)), long, steepener (−0.65/+0.9), flattener (+0.8/−0.6): identical to BCBS 368 | Delegated Regulation (EU) 2024/856 | Art. 2(1)–(4) | 2026-11-23 | `alm/curves.py::shock_vector` |
| 3 | LCR inflow cap | 75% of total expected outflows | BCBS LCR40 | LCR40.1, LCR40.77 | 2026-11-23 | module 3 |
| 3 | Level 1 assets: no haircut, unlimited share | 0% | BCBS LCR30 (effective 15 Dec 2019, chapter PDF) | LCR30.40–30.41 | 2026-11-23 | `config/alm.yaml` lcr.hqla |
| 3 | Level 2A haircut | 15% | BCBS LCR30 | LCR30.43 | 2026-11-23 | same |
| 3 | Level 2B corporate debt haircut | 50% | BCBS LCR30 | LCR30.45(2) | 2026-11-23 | same |
| 3 | Level 2 cap 40%, Level 2B cap 15% | caps after haircuts | BCBS LCR30 | LCR30.33, 30.34, 30.42 | 2026-11-23 | `alm/liquidity.py::hqla_stock` |
| 3 | Cap adjustments | Adj15 = max(L2B − 15/85·(L1+L2A), L2B − 15/60·L1, 0); Adj40 = max(L2A + L2B − Adj15 − 2/3·L1, 0); HQLA = L1 + L2A + L2B − Adj15 − Adj40 (the 30.38 total-stock formula is an image on the preceding page; the adjustments were read from the page image) | BCBS LCR30 | LCR30.38–30.39 | 2026-11-23 | same |
| 3 | Stable retail deposit run-off | 5% (3% national option not used) | BCBS LCR40 (chapter PDF) | LCR40.7, 40.11 | 2026-11-23 | `config/alm.yaml` lcr.outflow_rates |
| 3 | Less stable retail run-off | 10% minimum | BCBS LCR40 | LCR40.13 | 2026-11-23 | same |
| 3 | Operational deposits run-off | 25% | BCBS LCR40 | LCR40.26 | 2026-11-23 | same |
| 3 | Non-financial corporate non-operational run-off | 40% | BCBS LCR40 | LCR40.40 | 2026-11-23 | same |
| 3 | Unsecured wholesale from other legal entities (financial) | 100% | BCBS LCR40 | LCR40.42 | 2026-11-23 | same |
| 3 | Inflows from performing non-financial wholesale loans | 50% | BCBS LCR40 | LCR40.87(2) | 2026-11-23 | same |
| 3 | ASF factors | 100% capital and ≥1y liabilities; 95% stable retail; 90% less stable retail; 50% operational and non-financial corporate < 1y, other 6–12m funding; 0% other (< 6m from financials) | BCBS NSF30 (chapter PDF) | NSF30.10–30.14 | 2026-11-23 | `config/alm.yaml` nsfr.asf |
| 3 | RSF factors | 0% cash/central bank reserves; 5% Level 1 securities; 15% Level 2A; 50% Level 2B and non-HQLA < 1y; 65% ≥1y with RW ≤ 35%; 85% other performing loans ≥1y and non-HQLA securities ≥1y; 100% other assets | BCBS NSF30 | NSF30.25, 30.26, 30.28–30.32 | 2026-11-23 | `config/alm.yaml` nsfr.rsf |
| 3 | NMD core caps and average maturity | retail transactional 90% / 5y; retail non-transactional 70% / 4.5y; wholesale 50% / 4y; non-core overnight | BCBS 368 | paras 110–115, Table 2 | 2026-11-23 | `config/alm.yaml` deposits.bcbs_caps |
| 3 | EU NII large-decline threshold (Day 6) | decline of one-year NII by more than 5% of Tier 1 under either of the two NII scenarios (parallel up/down): (NII_shock − NII_base)/Tier 1 < −5% | Delegated Regulation (EU) 2024/856 | Art. 5(1)–(2) | 2026-11-23 | `config/alm.yaml` irrbb.nii_outlier_tier1 |
| 5 | Large exposure limit (Day 6) | exposure to a client or group of connected clients ≤ 25% of Tier 1 (the internal single-name limit of 15% of CET1 is stricter) | CRR | Art. 395(1) | 2026-11-23 | `docs/limits.md` |
| 3 | EU EVE outlier threshold (Day 6) | 15% of Tier 1 (as BCBS 368 para 88) | CRD | Art. 98(5) | 2026-11-23 | `config/alm.yaml` irrbb.eve_outlier_tier1 |
| 3 | EU deviations to note | EU treats extremely high-quality covered bonds as Level 1 with a 7% haircut; this project follows the Basel Level 2A treatment (15%). Not verified against the EU text | — | — | not verified | — |

**Day 7 (2026-11-23).** No new regulatory formulas. Module 7 tools, pinned in `uv.lock`: MLflow 3.18.0, Evidently 0.7.23, Dagster 1.13.31, typst (Python package) 0.15.2.
