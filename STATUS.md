# Status

## Day 1 (2026-11-22): foundation and synthetic bank — done
1. uv project (Python 3.12), ruff, mypy, pytest, pre-commit, Taskfile (go-task via uv dev dependency).
2. Seeded Polars generator: 50k obligors, 3 sectors x 4 grades = 12 segments, large-name tail; 16-line balance sheet with exact identity.
3. pandera contracts for loan book, balance sheet, rate curve; negative tests prove the checks fire.
4. ECB AAA spot curve 2020-01-02..2026-11-19 cached as parquet; pipeline reads only the cache.
5. `task test` 12 passed, `task lint` clean; content hashes locked in `data/synthetic/HASHES`; assumptions in `docs/data.md`.

## Day 2 (2026-11-23): module 1, credit portfolio model (GPU) — done
1. Multi-factor Gaussian copula (3 sector factors + global) and t-copula sensitivity in PyTorch; chunked, seeded per chunk; 2,000,000 scenarios on cuda.
2. 99.9% VaR 203.9m, ES 242.3m EUR; Euler ES contributions sum to ES exactly; EC = ES - EL per segment in results/credit/segment_capital.csv.
3. Cross-checks pass: homogeneous MC vs ASRF (CPU 6%, GPU 1.5% tolerance), Vasicek body quantiles, t-copula preserves marginal PD; one-factor real-book UL / ASRF = 1.022 (granularity add-on).
4. Benchmark (results/credit/benchmark.json): CUDA 177x numpy, 26x numba-parallel; energy per million scenarios recorded (RAPL + nvidia-smi).
5. CRE31.5 formulas and BCBS 368 shocks read from the BIS primary texts (EUR-Lex unreachable, decision 10); task test 21 passed, task lint clean.

## Day 3 (2026-11-23): module 2, Bayesian PD calibration — done
1. Seeded default history 2014-2025 (Vasicek year factor, macro index); time split train <= 2022, test 2023-2025; 4 low-default segments (2 with zero defaults).
2. NumPyro hierarchical model (sector prior, ordered grades, segment shrinkage, common year effect; 0 divergences): low-default mean |log error| 0.34 vs pooled 1.08; naive gives 2 zero PDs.
3. Truth recovery over 40 replications: coverage 0.57/0.86/0.94 at nominal 0.5/0.8/0.9; PPC p-values 0.47 and 0.59.
4. MAPIE CV+ conformal intervals for annual default rates (arcsine-sqrt scale): test coverage 0.89 at nominal 0.9.
5. Sector prior centres are assumptions (Destatis sector rates not retrievable, decision 16); task test 25 passed, task lint clean.

## Day 4 (2026-11-23): module 3, ALM, liquidity, IRRBB — done
1. Behavioural deposits: Weibull AFT (lifelines) on 24,000 simulated accounts, max |KM - model| 0.010; weighted core share 73% after BCBS 368 caps; 1y cumulative gap contractual -3.37bn vs behavioural 0.29bn EUR.
2. LCR 239%, NSFR 119%, with factors read from BCBS LCR30/LCR40/NSF30 (EU texts unreachable, decision 10).
3. Six BCBS 368 shocks: worst dEVE parallel_down -152.5m EUR = -21.3% of Tier 1 (EVE outlier: True); worst dNII parallel_down -13.1m EUR.
4. FTP curve (base + liquidity premium) with per-segment FTP charge and NIM after FTP in results/alm/ftp_segments.csv, the input for RAROC.
5. Tests: BCBS 368 worked examples reproduced exactly, zero shock gives zero delta, LCR/NSFR hand cases, ladders tie to the balance sheet; task test passes, task lint clean.

## Day 5 (2026-11-23): modules 4 and 5, RAROC, optimisation, limits — done
1. RAROC per segment after FTP, fees, opex and EL on Euler EC: bank 15.9% vs hurdle 10%; 3 of 12 below hurdle (manufacturing_1, manufacturing_4, trade_4); Euler allocation sums to the total.
2. cvxpy EVA maximisation under the capital budget, sector caps, growth bounds, LCR >= 130% and NSFR >= 110%: RAROC 15.9% -> 17.0% at unchanged capital (re-simulated, 2 passes); binding: capital_budget, growth_lower, growth_upper, sector_share_real_estate.
3. Limit system with traffic lights and escalation (docs/limits.md): {'amber': 1, 'green': 4, 'red': 2} before, {'amber': 3, 'green': 4, 'red': 0} after re-allocation, a 599m receive-fixed 5y hedge and an 11.3m single-name sell-down.
4. Tests: hand-solved LP (solution and shadow price), sector-limit LP, 40 random instances never violate constraints, allocation additivity and RAROC identity, hedge restores the EVE limit.
5. `task pipeline` (modules 1-5) runs end to end in about 3 minutes and reproduces the committed results; task test 44 passed, task lint clean. MVP (modules 1-5) complete.

## Day 6 (2026-11-23): module 6, LLM steering agent — done
1. Local agent (qwen3:14b via Ollama, JSON-schema constrained steps) with five allowlisted Pydantic tools (get_raroc, get_limits, run_shock, run_optimiser, search_rules) and a LanceDB/bge-m3 index over 931 stable paragraph chunks of CRR, CRD, RTS 2024/856, EBA/GL/2022/14, MaRisk, BCBS 368 and Basel LCR/NSFR/CRE chapters; MCP server exposes the same tools.
2. Deterministic verification before release: every number vs tool results (rounding-faithful), every citation vs retrieved chunks, polarity check for EVE/NII outlier claims, one-page limit; one repair round, otherwise refusal. Demo Vorlage in reports/vorstandsvorlage.md.
3. Eval (60 seeded questions, results/agent/eval_summary.json): answerable 24/24, multi-step 11/12 (1 blocked), out-of-scope refused 12/12, adversarial safe 12/12; first-attempt verification pass 92.5%, 211 released numbers all verified. Caveat: prompt and tools were tuned while looking at failures from this same set, so the rates are optimistic for unseen questions.
4. EU texts found locally: NII 5% threshold, EUR shock sizes and EVE 15% now verified (RTS 2024/856, CRD Art. 98(5)); EU post-shock floor implemented (non-binding, results unchanged). EU LCR regulation 2015/61 still missing.
5. task test 67 passed (LLM smoke test deselected), task lint clean. Known limits: citation relevance not checked; small integers can match unrelated tool values.
