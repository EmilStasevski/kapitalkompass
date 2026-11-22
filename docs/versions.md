# Regulatory formulas and their sources

Every regulatory formula or parameter used in code is recorded here with the primary text and
paragraph, read from the source document in this project (not recalled).

**Source access note (2026-11-23).** EUR-Lex (CRR 575/2013, Delegated Regulation 2015/61,
RTS 2024/856) answered automated requests with an empty HTTP 202 page, so the EU texts could
not be read in the automated run. The Basel Committee texts that the EU rules transpose were
read instead (BIS consolidated Basel Framework chapter PDFs and BCBS 368). Where the EU text is
known to differ or could not be confirmed, the row says so. See decision 10 in
`docs/decisions.md`.

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
| 3 | Post-shock floor | national discretion, floors not greater than zero | BCBS 368 | Annex 2, after (iv); footnote 46 | 2026-11-23 | module 3 |
| 3 | EVE outlier test | max ΔEVE over the six scenarios compared with 15% of Tier 1 | BCBS 368 | para 88 (Principle 12) | 2026-11-23 | module 3 |
| 3 | NII / additional outlier tests | at least as stringent as 15% of Tier 1; the EU NII threshold (RTS 2024/856) could not be read | BCBS 368 | para 89 | 2026-11-23 | module 3 |
| 3 | LCR inflow cap | 75% of total expected outflows | BCBS LCR40 | LCR40.1, LCR40.77 | 2026-11-23 | module 3 |
