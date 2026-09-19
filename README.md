# pytorch_metal_perf

MPS operator benchmarks across torch versions, shapes, layouts and dtypes.

The goal is to show whether PyTorch's Metal backend is actually getting faster
from 2.9 to 2.14 — and, just as importantly, to catch the cases where it got
faster on average while getting much slower for somebody.

## Why aggregates lie here

Most of the notable MPS performance work in this window was a migration: an op
moved off MPSGraph onto a hand-written Metal kernel, or onto
MetalPerformancePrimitives. These land with a benchmark table showing a large
win, and a good fraction of them also shipped a regression that a user found
later. A few from the record:

- **#152515** moved `mul` to TensorIterator for 2.8.0. A user bisected a 15–20%
  whole-application slowdown to it (**#168964**) and fixed it locally by
  un-registering `mul_stub` so it fell back to MPSGraph. Their diagnosis: the
  new kernel widens half inputs to fp32 for accuracy they did not need.
- **#189847**: `F.linear` on `[B, 1, K]` — the seq=1 LLM decode shape — ran up
  to **8.5x** slower than the mathematically identical 2D op on bf16/fp16. Cost
  ramped with `B`, then snapped back once `B` crossed ~16. fp32 was unaffected.
- **#185858**: Im2Col lost 12–41% on ordinary conv shapes because 64-bit index
  math, added to fix an overflow, sat in the hot path for every shape that
  never needed it.

None of these would have been caught by benchmarking one shape, one layout and
one dtype, and none of them would have shown up in a mean. So this suite sweeps
the axes those bugs actually lived on, and reports the tail next to the average.

### Failure patterns

Each is a sweep axis, and `regressions/` pins each to the PRs involved.

| | pattern | representative |
|---|---|---|
| **A** | launch overhead traded for kernel throughput — one cached MPSGraph becomes N Metal encoders | #152515 |
| **B** | precision widening: half inputs computed in fp32 | #152515, #189617 |
| **C** | index-width inflation: 64-bit math added for >2³² correctness | #185858, #185664, #188816 |
| **D** | lost specialization on non-dense inputs | #185291, #188483, #187313 |
| **E** | shape-regime cliff at a heuristic threshold | #189847, #186927, #189496 |
| **F** | memory-format assumption (channels_last) | #174241, #192551 |
| **G** | silent extra copy, usually `.contiguous()` on a grad path | #182714, #181951 |
| **H** | hidden CPU↔GPU sync inside the kernel | #195714, #191274 |
| **I** | misaligned storage offset | #195081, #186821 |
| **J** | scalar / broadcast fast-path divergence | #187229, #188560 |

## Method

**Two timing modes, and the gap between them is a signal.**

- `latency` — sync, call, sync. One op at a time, queue drained. What a user
  feels for a single eager op, including per-dispatch cost.
- `throughput` — submit N calls, sync once. The pipelined kernel cost with
  dispatch amortized away. This is the number quoted in "3x faster" PRs.

A large `latency / throughput` ratio means the wall-clock is host-side work or a
hidden sync, not kernel time. That distinction is not cosmetic. The seq=1 decode
bug shows 7.1x in throughput mode and 4.3x in latency mode on 2.9.0 — the
throughput view is the one that matches the 8.5x in the bug report, because the
cost really was in the kernel and latency mode dilutes it with a sync round trip
both paths pay. Elsewhere the ordering inverts: small matmuls on trunk are 10.9x
worse in latency than in throughput, which is pure dispatch overhead and not a
kernel problem at all. Either mode alone misreads one of these.
See [MEASUREMENTS.md](MEASUREMENTS.md).

**Two extremes per op.** `launch`-bound uses a 256×256 tensor, small enough that
per-encoder cost dominates. `memory`-bound uses 4096×4096, far past any cache, so
DRAM bandwidth dominates. Matmul adds a `compute` lane.

**Normalized against measured device peak.** Each environment probes achievable
bandwidth and FLOP/s with plain torch ops, and results carry `pct_peak_bw` /
`pct_peak_flops`. Raw milliseconds do not compare across an M1 and an M4 Pro;
"63% of achievable bandwidth" does. The probe is also a canary — it should not
vary with the torch version, and if it does, the machine changed state mid-sweep.

**Interleaved passes.** Versions run round-robin (v1,v2,v3, v1,v2,v3, …) rather
than one to completion. A laptop under sustained GPU load throttles, so a
version-at-a-time schedule systematically penalises whichever version ran last.
The reported number is a median across passes of each pass's own median.

**The OS build is part of the result identity.** macOS gates which kernel MPS
selects — MetalPerformancePrimitives availability is OS-dependent, MPSGraph
rewrites its own plans between releases, and PyTorch carries `is_macos_or_newer`
branches. A torch 2.9 run captured on macOS 15 against a torch 2.14 run on
macOS 26 measures the OS upgrade at least as much as it measures PyTorch. So
every run records the build (`25G83`, not just `26.6.2`) plus the versions of
Metal, MPS, MPSGraph and MetalPerformancePrimitives, results are filed under a
machine+OS slug, and `analyze.py` refuses to put two OS builds on one axis
unless you pass `--allow-mixed-env`.

> **If you upgrade macOS, re-run the whole matrix.** A partially re-baselined
> series is worse than no series, because the discontinuity looks like a torch
> regression.

## Usage

```bash
# one venv per torch version (~2.5 GB each), built on demand
python tools/sweep.py --setup-only

# full matrix, 3 interleaved passes
python tools/sweep.py

# faster: a representative subset
python tools/sweep.py --quick --passes 2

# one group, one dtype, against the local build only
python tools/sweep.py --versions local --groups matmul --dtypes bfloat16

# analysis (stdlib only — runs in any interpreter, no torch needed)
python tools/analyze.py                                   # text
python tools/analyze.py --format markdown --out REPORT.md
python tools/analyze.py --format json --out reports/data.json
python tools/report.py reports/data.json --out reports/index.html
```

A single environment can also be run directly:

```bash
PYTHONPATH=. python -m pmp.runner --out /tmp/run.jsonl --groups unary binary --quick
```

## Layout

```
pmp/
  harness.py     timing modes, adaptive iteration counts, in-flight budget
  device.py      machine + OS + framework fingerprint, measured device peaks
  layouts.py     tensor layout variants (the lost-fast-path detector)
  case.py        the unit of measurement
  runner.py      one sweep in one torch env -> JSONL
  ops/           one module per group
tools/
  sweep.py       multi-version orchestration, interleaved
  analyze.py     JSONL -> trends + regressions (stdlib only)
  report.py      JSON -> self-contained HTML
regressions/     one file per historical regression, PRs verified against tags
results/<machine+os>/<torch-version>/<timestamp>-p<pass>.jsonl
```

Results are JSONL with two record types: one `run` record per sweep carrying
machine, OS, peaks and torch build, then one `result` per (case, mode). Machine
metadata is written once and joined on `run_id` rather than stamped onto every
row.

## Coverage

Implemented: `overhead`, `unary`, `binary`, `reduction`, `normalization`,
`matmul`.

`overhead` is the calibration group — ops that do essentially no work, so the
other groups' `launch`-bound numbers can be read against a known floor. On an
M4 Pro that floor is ~170 µs for a sync round trip and ~12 µs per pipelined
encoder, while pure Python dispatch is 0.6 µs. Anything cheaper than the floor
is measuring the floor. See [MEASUREMENTS.md](MEASUREMENTS.md).

Not yet: scan/cumulative, indexing and gather/scatter, copy/cast, attention,
sort/topk, distributions, conv. Three entries in `regressions/`
(`im2col-64bit-index`, `d2h-nondense-gather`, `inductor-layout-channels-last`)
name cells in groups that do not exist yet and are therefore not detectable
today.

v1 is **eager-only**. The `torch.compile` / MPSInductor lane is not implemented,
so pattern F — of which #192551 is a 4.5–9x example — is currently out of reach.

Backward passes are not measured yet either, which leaves pattern G
(`.contiguous()` on a grad path) uncovered.
