# Does this thing actually detect anything?

A benchmark suite that has never caught a known bug is a hypothesis, not a tool.
Before adding breadth, the harness was pointed at one regression with a
documented magnitude, a documented shape dependence and a fix whose release is
verifiable from the git tags.

## Target: #189847, the seq=1 decode cliff

> `F.linear` on a 3D input of decode shape `[B, 1, K]` is up to ~8.5x slower
> than the mathematically identical 2D op `[B, K]`. The cost ramps ~linearly
> with `B` and then snaps back to GEMM once `B` crosses a threshold (~16).
> Specific to MPS, to `F.linear`, and to half precision — fp32 is unaffected.

Fixed by **#189855**, which first appears in the `v2.14.0` tag:

```
for t in v2.9.0 v2.10.0 v2.11.0 v2.12.1 v2.13.0 v2.14.0; do
  git log --oneline "$t" --grep='#189855)' -1 && break
done
```

So a working harness must show a large penalty on 2.9.0/2.11.0/2.13.0 that
disappears at 2.14.0, only at intermediate `B`, and only in half precision.

## Result

Apple M4 Pro (20 GPU cores), macOS 26.6.2 build 25G83, `K = N = 4096`, bf16,
2 interleaved passes. Ratio is `F.linear([B,1,K]) / F.linear([B,K])` — the same
arithmetic, so 1.00x is the correct answer and anything above it is overhead.

**Throughput mode** (pipelined, dispatch amortized):

| torch | B=1 | B=8 | B=17 |
|---|---:|---:|---:|
| 2.9.0 | 1.00x | **7.06x** | 1.02x |
| 2.11.0 | 0.99x | **7.29x** | 1.01x |
| 2.13.0 | 1.00x | **7.18x** | 1.05x |
| 2.14.0 | 1.01x | 1.02x | 1.01x |
| 2.15.0a0+git9a7fba2 | 0.96x | 1.01x | 1.02x |

**Latency mode** (sync per call):

| torch | B=1 | B=8 | B=17 |
|---|---:|---:|---:|
| 2.9.0 | 1.72x | **4.28x** | 1.02x |
| 2.11.0 | 0.92x | **4.33x** | 0.94x |
| 2.13.0 | 0.84x | **4.59x** | 1.16x |
| 2.14.0 | 0.90x | 0.93x | 1.02x |
| 2.15.0a0+git9a7fba2 | 0.80x | 0.83x | 1.25x |

Every property from the report reproduces: the magnitude (7.1x against a
reported "up to 8.5x" at a different `K`/`N`), the shape dependence (nothing at
`B=1`, nothing at `B=17`, the cliff in between), and the disappearance in
exactly the release the fix was verified into.

### Why both modes are kept

The two modes disagree about *how bad* this is, and the disagreement is the
useful part.

Here **throughput is the truthful mode**: 7.1x, matching the bug report. Latency
mode understates it at 4.3x, because both the 3D and 2D paths pay the same
~0.2 ms sync round trip and that constant dilutes the ratio.

Elsewhere the ordering inverts. On trunk, a 64×256×128 matmul is **10.9x** worse
in latency than in throughput (0.139 ms vs 0.013 ms). Nothing is wrong with that
kernel — the wall-clock is host-side dispatch. A throughput-only suite reports it
as healthy; a latency-only suite blames the kernel.

Neither mode is "the real number." The pair is the measurement.

## What the same run said about the harness itself

Two methodology bugs, both found by running it rather than by reading it:

**The peak probe drifted 13%.** Achievable fp32 FLOP/s fell 6534 → 5689 GFLOP/s
over a single five-version pass, bandwidth 205 → 196 GB/s. The probe measures
the machine, not torch, so all of that is thermal drift — and it bounds how
small a difference the sweep can honestly resolve. `analyze.py` now computes
this spread and prints a warning when it exceeds 5%, because a 1.13x geomean
sitting on top of 13% drift is not a result.

**Interleaving was not enough.** Passes ran `v1,v2,v3,v4,v5` every time, so the
first version always measured on a cool GPU and the last always on a hot one —
converting drift into a systematic bias against later versions rather than
averaging it away. The pass order is now rotated, and `--cooldown` (default 15 s)
idles between runs.

A third, smaller one: `p10` was computed by index arithmetic that collapses onto
the minimum for small samples, so the "tail" column was silently a duplicate of
"worst" exactly when a group had few cells. It now interpolates.

## Caveats

- Single machine, single OS build. Everything here is conditional on macOS
  26.6.2 (25G83); see the OS-pinning note in the README.
- `bytes_moved` is a model, not an instrumented count. For ops that read their
  input more than once (softmax's two passes) or benefit from cache residency,
  achieved GB/s can exceed the measured DRAM peak — that is information about
  cache behaviour, not an error, but it means GB/s is a comparison aid rather
  than a hardware-counter truth.
- The regression corpus has nine entries; only this one has been reproduced end
  to end so far. The others name the cell that should expose them, and three
  name groups that do not exist yet.
