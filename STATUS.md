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
