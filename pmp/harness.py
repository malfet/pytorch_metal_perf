"""Timing primitives for MPS.

Two modes, because on MPS they answer different questions and the gap between
them is itself a signal:

  latency     sync, call, sync. One op at a time, queue drained. This is what a
              user feels for a single eager op, and it includes the per-dispatch
              cost that MPSGraph->Metal migrations tend to inflate.

  throughput  submit N calls, sync once. The pipelined cost of the kernel with
              dispatch overhead amortized away. This is the number that gets
              quoted in "we made it 3x faster" PRs.

latency/throughput ratio is the hidden-sync detector: a kernel that internally
blocks on the CPU (a per-element scratch readback, a per-bin sync) cannot
pipeline, so its throughput time never drops below its latency time.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch

# How much in-flight output we are willing to let the allocator hold between
# syncs. Without this bound, a 64 MiB-output op submitted 200 times unsynced
# asks the caching allocator for 12 GiB, and we end up timing memory pressure.
INFLIGHT_BUDGET_BYTES = 512 << 20

# Target wall time for one measured repetition. Long enough to swamp the ~50us
# sync, short enough that a full sweep finishes.
TARGET_REP_S = 0.030
MIN_REP_S = 0.005


def sync() -> None:
    torch.mps.synchronize()


@dataclass
class TimingResult:
    median_s: float
    mean_s: float
    min_s: float
    iqr_s: float
    reps: int
    iters_per_rep: int
    iters_per_sync: int
    mode: str

    @property
    def rel_iqr(self) -> float:
        """IQR as a fraction of the median -- the per-case noise estimate."""
        return self.iqr_s / self.median_s if self.median_s > 0 else float("inf")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rel_iqr"] = self.rel_iqr
        return d


def _summarize(per_call: list[float], reps: int, iters: int, chunk: int, mode: str) -> TimingResult:
    s = sorted(per_call)
    if len(s) >= 4:
        q1, q3 = statistics.quantiles(s, n=4, method="inclusive")[0], statistics.quantiles(
            s, n=4, method="inclusive"
        )[2]
        iqr = q3 - q1
    else:
        iqr = (s[-1] - s[0]) if len(s) > 1 else 0.0
    return TimingResult(
        median_s=statistics.median(s),
        mean_s=statistics.fmean(s),
        min_s=s[0],
        iqr_s=iqr,
        reps=reps,
        iters_per_rep=iters,
        iters_per_sync=chunk,
        mode=mode,
    )


def _calibrate(fn, warmup: int) -> float:
    """Warm up (compiling shaders, populating the MPSGraph cache) and return a
    rough single-call cost used to size the real measurement."""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(3):
        fn()
    sync()
    return max((time.perf_counter() - t0) / 3, 1e-7)


def time_latency(fn, *, reps: int = 9, warmup: int = 8) -> TimingResult:
    """Sync-per-call. Each sample is one full submit->complete round trip."""
    _calibrate(fn, warmup)
    samples: list[float] = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return _summarize(samples, reps, 1, 1, "latency")


def time_throughput(
    fn,
    *,
    out_bytes: int = 0,
    reps: int = 7,
    warmup: int = 8,
    min_iters: int = 1,
    target_s: float = TARGET_REP_S,
) -> TimingResult:
    """Submit many calls per sync, so per-dispatch cost is amortized.

    `out_bytes` is the size of the tensor the op allocates per call; it bounds
    how many calls we dare leave in flight. Pass 0 for ops that write into a
    preallocated output.
    """
    per_call = _calibrate(fn, warmup)

    chunk = INFLIGHT_BUDGET_BYTES // out_bytes if out_bytes > 0 else 1024
    chunk = max(min_iters, min(chunk, 1024))

    iters = max(chunk, int(target_s / per_call))
    iters = (iters // chunk) * chunk or chunk
    iters = min(iters, 200_000)

    samples: list[float] = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        done = 0
        while done < iters:
            for _ in range(chunk):
                fn()
            sync()
            done += chunk
        elapsed = time.perf_counter() - t0
        samples.append(elapsed / done)
    return _summarize(samples, reps, iters, chunk, "throughput")


def measure(fn, mode: str, *, out_bytes: int = 0, **kw) -> TimingResult:
    if mode == "latency":
        return time_latency(fn, **{k: v for k, v in kw.items() if k in ("reps", "warmup")})
    if mode == "throughput":
        return time_throughput(fn, out_bytes=out_bytes, **kw)
    raise ValueError(f"unknown timing mode: {mode!r}")
