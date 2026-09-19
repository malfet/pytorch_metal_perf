# pytorch_metal_perf

**Machine:** Apple M4 Pro / Apple M4 Pro (20 GPU cores)
**OS:** macOS 26.6.2 build 25G83
**MetalPerformancePrimitives:** 1
**Baseline:** torch 2.9.0

## Measured device peaks (sanity check -- should not vary with torch)

| torch | bandwidth GB/s | fp32 GFLOP/s | fp16 GFLOP/s |
|---|---:|---:|---:|
| 2.9.0 | 223 | 6769 | 7492 |
| 2.11.0 | 222 | 6775 | 7485 |
| 2.13.0 | 214 | 6604 | 7213 |
| 2.14.0 | 202 | 6659 | 7361 |
| 2.15.0a0+git9a7fba2 | 213 | 6582 | 7432 |

> **Note.** Peak probe varied by 10% (bandwidth_gbps), 4% (gflops_fp16), 3% (gflops_fp32) across versions. That exceeds 5%: the GPU was throttling, so differences below roughly this magnitude are drift, not torch. Raise --cooldown or --passes and re-run.

## Speedup vs torch 2.9.0 (higher is better)

### latency

| group | stat | 2.9.0 | 2.11.0 | 2.13.0 | 2.14.0 | 2.15.0a0+git9a7fba2 |
|---|---|---:|---:|---:|---:|---:|
| binary (n=96) | geomean | 1.00x | 1.08x | 1.07x | 1.03x | 1.02x |
|  | p10 | 1.00x | 0.96x | 0.91x | 0.87x | 0.82x |
|  | worst | 1.00x | 0.82x | 0.72x | 0.69x | 0.56x |
| matmul (n=42) | geomean | 1.00x | 1.02x | 0.94x | 1.09x | 1.10x |
|  | p10 | 1.00x | 0.95x | 0.91x | 0.95x | 0.94x |
|  | worst | 1.00x | 0.87x | 0.27x | 0.89x | 0.90x |
| normalization (n=54) | geomean | 1.00x | 1.02x | 1.11x | 1.11x | 1.08x |
|  | p10 | 1.00x | 0.93x | 0.93x | 0.94x | 0.90x |
|  | worst | 1.00x | 0.81x | 0.86x | 0.86x | 0.73x |
| overhead (n=7) | geomean | 1.00x | 1.19x | 1.25x | 1.36x | 1.34x |
|  | p10 | 1.00x | 0.93x | 0.92x | 0.93x | 0.86x |
|  | worst | 1.00x | 0.87x | 0.81x | 0.93x | 0.77x |
| reduction (n=108) | geomean | 1.00x | 1.03x | 1.13x | 1.45x | 1.43x |
|  | p10 | 1.00x | 0.95x | 0.90x | 1.01x | 1.02x |
|  | worst | 1.00x | 0.86x | 0.48x | 0.56x | 0.60x |
| unary (n=72) | geomean | 1.00x | 1.04x | 1.18x | 1.15x | 1.16x |
|  | p10 | 1.00x | 0.95x | 0.90x | 0.91x | 0.85x |
|  | worst | 1.00x | 0.77x | 0.71x | 0.68x | 0.69x |

### throughput

| group | stat | 2.9.0 | 2.11.0 | 2.13.0 | 2.14.0 | 2.15.0a0+git9a7fba2 |
|---|---|---:|---:|---:|---:|---:|
| binary (n=96) | geomean | 1.00x | 1.21x | 1.14x | 1.10x | 1.14x |
|  | p10 | 1.00x | 0.99x | 0.90x | 0.91x | 0.80x |
|  | worst | 1.00x | 0.86x | 0.60x | 0.46x | 0.50x |
| matmul (n=42) | geomean | 1.00x | 1.01x | 0.90x | 1.09x | 1.09x |
|  | p10 | 1.00x | 0.95x | 0.91x | 0.95x | 0.92x |
|  | worst | 1.00x | 0.93x | 0.20x | 0.89x | 0.85x |
| normalization (n=54) | geomean | 1.00x | 1.00x | 1.10x | 1.10x | 1.15x |
|  | p10 | 1.00x | 0.93x | 0.91x | 0.92x | 0.93x |
|  | worst | 1.00x | 0.86x | 0.88x | 0.84x | 0.87x |
| overhead (n=7) | geomean | 1.00x | 0.96x | 1.15x | 1.52x | 1.46x |
|  | p10 | 1.00x | 0.90x | 0.85x | 0.88x | 0.82x |
|  | worst | 1.00x | 0.78x | 0.77x | 0.86x | 0.72x |
| reduction (n=108) | geomean | 1.00x | 1.01x | 1.39x | 1.99x | 2.07x |
|  | p10 | 1.00x | 0.96x | 0.94x | 1.01x | 1.02x |
|  | worst | 1.00x | 0.85x | 0.39x | 0.49x | 0.48x |
| unary (n=72) | geomean | 1.00x | 1.06x | 1.26x | 1.21x | 1.19x |
|  | p10 | 1.00x | 0.85x | 0.81x | 0.80x | 0.70x |
|  | worst | 1.00x | 0.71x | 0.56x | 0.52x | 0.60x |

## Regressions vs previous version

| slowdown | from | to | group | op | variant | dtype | bound | mode | before ms | after ms |
|---:|---|---|---|---|---|---|---|---|---:|---:|
| **5.34x** | 2.11.0 | 2.13.0 | matmul | mv | 4096x4096 | float16 | memory | throughput | 0.148 | 0.789 |
| **5.08x** | 2.11.0 | 2.13.0 | matmul | mv | 11008x4096 | float16 | memory | throughput | 0.386 | 1.961 |
| **4.30x** | 2.11.0 | 2.13.0 | matmul | mv | 11008x4096 | float16 | memory | latency | 0.504 | 2.166 |
| **2.86x** | 2.11.0 | 2.13.0 | matmul | mv | 4096x4096 | float16 | memory | latency | 0.332 | 0.950 |
| **2.36x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | bfloat16 | memory | throughput | 0.380 | 0.899 |
| **2.31x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | float16 | memory | throughput | 0.377 | 0.871 |
| **2.29x** | 2.11.0 | 2.13.0 | binary | mul | dense-dense | float32 | launch | throughput | 0.003 | 0.006 |
| **2.12x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | float16 | memory | latency | 0.513 | 1.089 |
| **2.10x** | 2.13.0 | 2.14.0 | reduction | max | inner_slice/full | float16 | memory | throughput | 0.292 | 0.614 |
| **2.03x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | bfloat16 | memory | latency | 0.510 | 1.037 |
| **2.00x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | dense-scalar | float32 | launch | latency | 0.100 | 0.199 |
| **1.76x** | 2.13.0 | 2.14.0 | reduction | max | inner_slice/full | bfloat16 | memory | throughput | 0.330 | 0.583 |
| **1.74x** | 2.13.0 | 2.14.0 | reduction | max | inner_slice/full | float32 | memory | throughput | 0.661 | 1.148 |
| **1.65x** | 2.13.0 | 2.14.0 | binary | div | transp-transp | bfloat16 | launch | throughput | 0.003 | 0.006 |
| **1.60x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | float32 | memory | throughput | 0.593 | 0.947 |
| **1.60x** | 2.13.0 | 2.14.0 | reduction | max | inner_slice/full | bfloat16 | memory | latency | 0.547 | 0.872 |
| **1.59x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | islice-islice | float16 | memory | throughput | 0.750 | 1.190 |
| **1.57x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/full | float16 | memory | throughput | 0.315 | 0.495 |
| **1.55x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/full | bfloat16 | memory | throughput | 0.325 | 0.504 |
| **1.53x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | dense-scalar | float16 | launch | latency | 0.099 | 0.152 |
| **1.52x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | islice-islice | bfloat16 | memory | throughput | 0.785 | 1.190 |
| **1.51x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | dense-scalar | bfloat16 | launch | latency | 0.095 | 0.144 |
| **1.49x** | 2.11.0 | 2.13.0 | unary | sqrt | transposed | bfloat16 | launch | latency | 0.116 | 0.173 |
| **1.49x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | islice-islice | float16 | memory | latency | 0.911 | 1.359 |
| **1.45x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/outer | float32 | memory | latency | 0.761 | 1.100 |
| **1.44x** | 2.13.0 | 2.14.0 | reduction | max | inner_slice/full | float32 | memory | latency | 0.975 | 1.404 |
| **1.44x** | 2.11.0 | 2.13.0 | reduction | sum | inner_slice/full | bfloat16 | memory | latency | 0.499 | 0.716 |
| **1.43x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | pow | islice-islice | bfloat16 | memory | latency | 0.944 | 1.353 |
| **1.28x** | 2.9.0 | 2.11.0 | overhead | view | floor | float32 | launch | throughput | 0.000 | 0.000 |
| **1.23x** | 2.14.0 | 2.15.0a0+git9a7fba2 | unary | exp | transposed | float16 | launch | throughput | 0.004 | 0.005 |
| **1.21x** | 2.14.0 | 2.15.0a0+git9a7fba2 | binary | div | islice-islice | float16 | launch | latency | 0.136 | 0.164 |
| **1.20x** | 2.13.0 | 2.14.0 | overhead | empty_like | floor | float32 | launch | throughput | 0.000 | 0.000 |
| **1.19x** | 2.13.0 | 2.14.0 | reduction | sum | dense/inner | float16 | launch | throughput | 0.003 | 0.004 |
| **1.16x** | 2.11.0 | 2.13.0 | reduction | max | inner_slice/full | float32 | memory | throughput | 0.572 | 0.661 |

## Dispatch-bound cells (latency >= 3x pipelined cost, torch 2.15.0a0+git9a7fba2)

Host-side overhead or a hidden sync, not kernel time. Invisible to a throughput-only benchmark.

| ratio | group | op | variant | dtype | bound | latency ms | pipelined ms |
|---:|---|---|---|---|---|---:|---:|
| 56.5x | overhead | neg_32x32 | floor | float32 | launch | 0.112 | 0.002 |
| 48.5x | unary | neg | dense | float16 | launch | 0.135 | 0.003 |
| 47.5x | unary | neg | dense | float32 | launch | 0.137 | 0.003 |
| 47.4x | binary | mul | dense-dense | bfloat16 | launch | 0.134 | 0.003 |
| 46.1x | unary | neg | transposed | float32 | launch | 0.136 | 0.003 |
| 45.7x | overhead | mm_1x1 | floor | float32 | launch | 0.134 | 0.003 |
| 45.4x | unary | neg | transposed | float16 | launch | 0.128 | 0.003 |
| 42.2x | binary | mul | transp-transp | bfloat16 | launch | 0.118 | 0.003 |
| 40.2x | unary | erf | dense | bfloat16 | launch | 0.162 | 0.004 |
| 40.0x | unary | exp | dense | float16 | launch | 0.186 | 0.005 |
| 39.0x | unary | neg | dense | bfloat16 | launch | 0.203 | 0.005 |
| 39.0x | reduction | sum | dense/inner | bfloat16 | launch | 0.137 | 0.004 |
| 38.8x | binary | mul | dense-dense | float32 | launch | 0.155 | 0.004 |
| 37.6x | binary | div | dense-dense | float32 | launch | 0.134 | 0.004 |
| 36.9x | unary | sqrt | transposed | bfloat16 | launch | 0.166 | 0.004 |
| 36.6x | overhead | add_1elem | floor | float32 | launch | 0.114 | 0.003 |
| 36.5x | binary | add | transp-transp | float16 | launch | 0.160 | 0.004 |
| 36.4x | reduction | argmax | dense/inner | float16 | launch | 0.134 | 0.004 |
| 36.4x | binary | mul | transp-transp | float16 | launch | 0.147 | 0.004 |
| 36.2x | reduction | sum | dense/inner | float32 | launch | 0.142 | 0.004 |
| 36.2x | unary | exp | dense | bfloat16 | launch | 0.190 | 0.005 |
| 36.1x | unary | erf | transposed | bfloat16 | launch | 0.150 | 0.004 |
| 35.9x | binary | mul | transp-transp | float32 | launch | 0.148 | 0.004 |
| 35.9x | unary | exp | transposed | float32 | launch | 0.171 | 0.005 |
| 35.3x | reduction | argmax | dense/inner | float32 | launch | 0.128 | 0.004 |
