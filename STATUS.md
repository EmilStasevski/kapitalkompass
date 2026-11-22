# Status

## Day 1 (2026-11-22): foundation and synthetic bank — done
1. uv project (Python 3.12), ruff, mypy, pytest, pre-commit, Taskfile (go-task via uv dev dependency).
2. Seeded Polars generator: 50k obligors, 3 sectors x 4 grades = 12 segments, large-name tail; 16-line balance sheet with exact identity.
3. pandera contracts for loan book, balance sheet, rate curve; negative tests prove the checks fire.
4. ECB AAA spot curve 2020-01-02..2026-11-19 cached as parquet; pipeline reads only the cache.
5. `task test` 12 passed, `task lint` clean; content hashes locked in `data/synthetic/HASHES`; assumptions in `docs/data.md`.
