"""Benchmark: numpy (1 core) vs numba (parallel) vs torch CPU vs torch CUDA.

Same model and portfolio, fixed (scenario x obligor) workload. Energy: GPU board power
sampled with nvidia-smi (100 ms), CPU package energy from Intel RAPL. Hardware-dependent,
so it is a separate task (`task bench`), not part of the reproducible pipeline.
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
import numpy as np
import polars as pl
import torch
from scipy.stats import norm

from kapitalkompass.config import SYNTH_DIR, load_config
from kapitalkompass.credit.portfolio import Portfolio, simulate
from kapitalkompass.credit.run import build_portfolio
from kapitalkompass.results import write_json

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")


def _prep(pf: Portfolio) -> tuple[np.ndarray, ...]:
    a = np.sqrt(pf.rho_sector[pf.sector])
    b = np.sqrt(1 - pf.rho_sector[pf.sector])
    c = norm.ppf(pf.pd)
    return a, b, c, pf.ead * pf.lgd, pf.sector


def sim_numpy(pf: Portfolio, scenarios: int, seed: int, chunk: int = 500) -> np.ndarray:
    a, b, c, w, sec = _prep(pf)
    rng = np.random.default_rng(seed)
    out = np.empty(scenarios)
    sw, sw1 = np.sqrt(pf.global_weight), np.sqrt(1 - pf.global_weight)
    for st in range(0, scenarios, chunk):
        m = min(chunk, scenarios - st)
        z = sw * rng.standard_normal((m, 1)) + sw1 * rng.standard_normal((m, 3))
        eps = rng.standard_normal((m, pf.n))
        lat = a * z[:, sec] + b * eps
        out[st : st + m] = (lat < c) @ w
    return out


@numba.njit(parallel=True, fastmath=False, cache=True)
def _numba_kernel(a, b, c, w, sec, gw, scenarios, seed):
    out = np.empty(scenarios)
    sw, sw1 = np.sqrt(gw), np.sqrt(1 - gw)
    n = a.shape[0]
    for s in numba.prange(scenarios):
        np.random.seed(seed + s)
        g = np.random.standard_normal()
        z = np.empty(3)
        for k in range(3):
            z[k] = sw * g + sw1 * np.random.standard_normal()
        loss = 0.0
        for i in range(n):
            if a[i] * z[sec[i]] + b[i] * np.random.standard_normal() < c[i]:
                loss += w[i]
        out[s] = loss
    return out


def sim_numba(pf: Portfolio, scenarios: int, seed: int) -> np.ndarray:
    a, b, c, w, sec = _prep(pf)
    return _numba_kernel(a, b, c, w, sec, pf.global_weight, scenarios, seed)


def _cuda_run(pf: Portfolio, scenarios: int) -> None:
    simulate(pf, scenarios, 1, "cuda")
    torch.cuda.synchronize()


class PowerMeter:
    def __init__(self) -> None:
        self.gpu_w: list[float] = []
        self._stop = threading.Event()

    def _gpu_loop(self) -> None:
        p = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert p.stdout is not None
        while not self._stop.is_set():
            line = p.stdout.readline()
            if line.strip():
                self.gpu_w.append(float(line))
        p.terminate()

    def __enter__(self) -> PowerMeter:
        self.e0 = int(RAPL.read_text()) if RAPL.exists() else None
        self.t0 = time.perf_counter()
        self.th = threading.Thread(target=self._gpu_loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.t0
        self._stop.set()
        self.th.join(timeout=2)
        self.cpu_j: float | None = None
        if self.e0 is not None:
            d = int(RAPL.read_text()) - self.e0
            if d < 0:
                d += int(RAPL_MAX.read_text())
            self.cpu_j = d / 1e6
        self.gpu_j = float(np.mean(self.gpu_w)) * self.seconds if self.gpu_w else None


def measure(fn: Callable[[], Any]) -> dict[str, Any]:
    fn_warm = fn
    with PowerMeter() as pm:
        fn_warm()
    return {"seconds": pm.seconds, "cpu_package_j": pm.cpu_j, "gpu_board_j": pm.gpu_j}


def main(scen_small: int = 5000, scen_large: int = 1_000_000) -> None:
    cfg = load_config("model")["credit"]
    pf = build_portfolio(pl.read_parquet(SYNTH_DIR / "loan_book.parquet"), cfg)
    sim_numba(pf, 8, 0)  # JIT compile outside timing
    simulate(pf, 1000, 0, "cuda")  # CUDA context + kernels warm
    rows = []
    runs: list[tuple[str, int, Callable[[], Any]]] = [
        ("numpy_1core", scen_small, lambda: sim_numpy(pf, scen_small, 1)),
        ("numba_parallel", scen_small * 8, lambda: sim_numba(pf, scen_small * 8, 1)),
        ("torch_cpu", scen_small * 8, lambda: simulate(pf, scen_small * 8, 1, "cpu", 2000)),
        (
            "torch_cuda",
            scen_large,
            lambda: _cuda_run(pf, scen_large),
        ),
    ]
    for name, s, fn in runs:
        m = measure(fn)
        pairs = s * pf.n
        total_j = (m["cpu_package_j"] or 0.0) + (m["gpu_board_j"] or 0.0)
        rows.append(
            {
                "engine": name,
                "scenarios": s,
                "obligors": pf.n,
                **m,
                "pairs_per_second": pairs / m["seconds"],
                "seconds_per_million_scenarios": m["seconds"] / s * 1e6,
                "joules_per_million_scenarios": total_j / s * 1e6,
            }
        )
        print(rows[-1])
    base = rows[0]["seconds_per_million_scenarios"]
    for r in rows:
        r["speedup_vs_numpy"] = base / r["seconds_per_million_scenarios"]
    write_json(
        "credit/benchmark.json",
        {
            "hardware": {
                "cpu": platform.processor() or "see lscpu",
                "cpu_model": _cpu_model(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "numba": numba.__version__,
            },
            "note": "energy = CPU package (RAPL) + GPU board (nvidia-smi mean power x time), "
            "idle draw included; timings exclude JIT/CUDA warm-up",
            "runs": rows,
        },
    )


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


if __name__ == "__main__":
    main()
