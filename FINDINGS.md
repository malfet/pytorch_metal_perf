# Findings — first full sweep

Apple M4 Pro (20 GPU cores), macOS 26.6.2 build 25G83, torch 2.9.0 / 2.11.0 /
2.13.0 / 2.14.0 / 2.15.0a0+git9a7fba2, 3 interleaved passes, `--quick` subset
(372 cases × 2 timing modes per version-pass).

Peak-probe drift across the sweep was 10% on bandwidth and 3–4% on FLOP/s, so
treat anything under ~1.15x as noise. Every finding below is well outside that
and was re-confirmed with a standalone script, independent of the harness.

## The trend

Throughput-mode geomean vs 2.9.0:

| group | 2.11.0 | 2.13.0 | 2.14.0 | trunk |
|---|---:|---:|---:|---:|
| reduction | 1.01x | 1.39x | **1.99x** | 2.07x |
| unary | 1.06x | 1.26x | 1.21x | 1.19x |
| overhead (dispatch floor) | 0.96x | 1.15x | 1.52x | 1.46x |
| normalization | 1.00x | 1.10x | 1.10x | 1.15x |
| binary | 1.21x | 1.14x | 1.10x | 1.14x |
| matmul | 1.01x | 0.90x | 1.09x | 1.09x |

Reductions doubled at 2.14.0, exactly where the #191097–#191104 series landed
(verified against the `v2.14.0` tag). The dispatch floor itself improved 1.5x,
which lifts every small-tensor op regardless of kernel quality.

Now the `worst` column for that same doubled reduction group: **0.49x**. That
gap is the entire reason this repo exists.

## 1. `torch.mv` on fp16 was 4.3x slower in 2.13.0 — and only in 2.13.0

Independent confirmation, median of 9 × 100 calls:

| torch | mv 4096×4096 fp16 | mv 11008×4096 fp16 | fp32 control |
|---|---:|---:|---:|
| 2.11.0 | 0.170 ms | 0.403 ms | 0.311 / 0.771 ms |
| 2.13.0 | **0.733 ms** | **1.813 ms** | 0.315 / 0.778 ms |
| 2.14.0 | 0.166 ms | 0.382 ms | 0.311 / 0.755 ms |

fp32 is flat across all three, so this is half-precision only.

2.13.0 explains itself — it emits:

> `UserWarning: MPS mm implementation has a known issue with this shape, dtype
> and slice. Dispatching to metal implementation instead. This may impact
> performance.` — `LinearAlgebra.mm`

The guard is the LORADOWN GEMV padding-overflow workaround for #178056: fp16,
`self.size(0) <= 16 || other.size(1) <= 16`, unit strides. A matrix-vector
product has `other.size(1) == 1`, so **every fp16 GEMV takes it**. In 2.13.0 the
metal fallback was ~4.3x slower than the MPSGraph path it replaced; by 2.14.0
the dedicated GEMV kernels (#186927) made the fallback fast, so the same guard
now costs nothing.

The correctness fix was right. The cost of the fallback it selected was not
measured at the time, and one release shipped with fp16 decode GEMV 4.3x slow.

## 2. `max` over a strided view regressed 2.0x at 2.14.0 — still present on trunk

| torch | `max` on `x[:, ::2]` fp16 | `max` on dense fp16 |
|---|---:|---:|
| 2.11.0 | 0.288 ms | 0.157 ms |
| 2.13.0 | 0.289 ms | 0.164 ms |
| 2.14.0 | **0.582 ms** | 0.145 ms |
| trunk | **0.576 ms** | 0.147 ms |

The dense path is untouched — slightly better, in fact. Only the non-dense path
regressed, and it landed in the same release that doubled the reduction group's
geomean. Strided `max` is now **3.9x** the cost of dense over the same number of
elements.

This is pattern D exactly: a rewrite optimises the dense case and the general
case falls back to something worse. It is the single clearest live example of
why the geomean cannot be the only number reported, and unlike finding 1 it has
not been fixed.

## 3. `pow` on strided views regressed 1.6x on trunk, unreleased

| torch | `pow` on `x[:, ::2]` fp16 | bf16 |
|---|---:|---:|
| 2.14.0 | 0.740 ms | 0.742 ms |
| trunk | **1.173 ms** | **1.162 ms** |

Also visible at the launch-bound shape (`pow` with a scalar, fp32: 0.006 →
0.009 ms). Between 2.14.0 and `2.15.0a0+git9a7fba2` only.

This one has not shipped yet, which makes it the most actionable of the three.

## Caveats

- One machine, one OS build, one `--quick` sweep of three passes. Findings 1–3
  were re-confirmed with a standalone script but not on a second machine.
- `--quick` samples a subset of ops and layouts; the full sweep is ~3.8x larger
  and may surface more.
- Bisection to a specific PR was only done for finding 1, and there only because
  torch named its own guard in a warning. Findings 2 and 3 are pinned to a
  release interval, not to a commit.
