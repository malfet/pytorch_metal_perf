"""Matmul / linear.

This group is shape-cliff country (pattern E). MPS picks between MPSGraph, a
hand-written GEMV, and MetalPerformancePrimitives tiles on thresholds, and a
shape sitting just on the wrong side falls off a cliff that a power-of-two
sweep will never see. Two documented examples drive the shape list:

  #189847  F.linear on [B, 1, K] (seq=1 decode) ran up to 8.5x slower than the
           identical 2D op on bf16/fp16, cost ramping with B and snapping back
           once B crossed ~16. Fixed by #189855 (2026-07-20 -> 2.14.0).
  #189496  F.linear with bias corrupted rows once a batch dim exceeded 2^16.

So batch is swept at 1,2,4,8,16,17,32 rather than powers of two alone, and the
2D form of every decode shape is measured alongside as the reference the 3D form
should match.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..case import Case, SweepConfig

#: (label, M, K, N). Mixed aligned / deliberately-unaligned, from the tile-size
#: tuning sweep in the MPP matmul2d gist.
GEMM_SHAPES = (
    ("llama-qkv", 2048, 4096, 4096),
    ("gpt2-ffn", 1024, 768, 3072),
    ("bert-proj", 4096, 768, 768),
    ("vit-patch", 197, 768, 768),      # unaligned M -- edge tiles
    ("seq-1023", 1023, 4096, 4096),    # one short of a tile boundary
    ("both-odd", 197, 513, 195),
    ("small-2d", 64, 256, 128),
)
QUICK_GEMM = ("llama-qkv", "vit-patch", "small-2d")

#: Batch values for the seq=1 decode probe. 16/17 straddle the documented cliff.
DECODE_BATCHES = (1, 2, 4, 8, 16, 17, 32)
QUICK_DECODE = (1, 8, 17)

DECODE_K, DECODE_N = 4096, 4096

_FLOAT = ("float32", "float16", "bfloat16")


def _gemm_cases(cfg: SweepConfig) -> Iterator[Case]:
    shapes = [s for s in GEMM_SHAPES if not cfg.quick or s[0] in QUICK_GEMM]
    for label, m, k, n in shapes:
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            a = torch.empty(m, k, dtype=dtype, device="mps").uniform_(-1, 1)
            b = torch.empty(k, n, dtype=dtype, device="mps").uniform_(-1, 1)
            esize = a.element_size()
            yield Case(
                group="matmul",
                op="mm",
                variant=label,
                dtype=dtype_name,
                bound="compute",
                shape=(m, k, n),
                fn=(lambda x=a, y=b: x @ y),
                bytes_moved=(m * k + k * n + m * n) * esize,
                flops=2 * m * k * n,
                out_bytes=m * n * esize,
                modes=cfg.modes,
                tags={"aligned": m % 64 == 0 and n % 64 == 0 and k % 64 == 0},
            )
            # Transposed B is what nn.Linear actually hands the backend.
            bt = torch.empty(n, k, dtype=dtype, device="mps").uniform_(-1, 1)
            yield Case(
                group="matmul",
                op="mm_bt",
                variant=label,
                dtype=dtype_name,
                bound="compute",
                shape=(m, k, n),
                fn=(lambda x=a, y=bt: x @ y.t()),
                bytes_moved=(m * k + k * n + m * n) * esize,
                flops=2 * m * k * n,
                out_bytes=m * n * esize,
                modes=cfg.modes,
            )


def _decode_cases(cfg: SweepConfig) -> Iterator[Case]:
    """The seq=1 decode cliff: F.linear on [B,1,K] vs the identical 2D [B,K]."""
    k, n = DECODE_K, DECODE_N
    batches = [b for b in DECODE_BATCHES if not cfg.quick or b in QUICK_DECODE]
    for dtype_name in cfg.dtypes:
        if dtype_name not in _FLOAT:
            continue
        dtype = getattr(torch, dtype_name)
        w = torch.empty(n, k, dtype=dtype, device="mps").uniform_(-1, 1)
        esize = w.element_size()
        for b in batches:
            x3 = torch.empty(b, 1, k, dtype=dtype, device="mps").uniform_(-1, 1)
            x2 = torch.empty(b, k, dtype=dtype, device="mps").uniform_(-1, 1)
            # Weight streaming dominates: B*K and B*N are negligible next to K*N.
            nbytes = (k * n + b * k + b * n) * esize
            flops = 2 * b * k * n
            common = dict(
                group="matmul",
                dtype=dtype_name,
                bound="memory",     # decode is weight-bandwidth-bound, not FLOP-bound
                shape=(b, 1, k),
                bytes_moved=nbytes,
                flops=flops,
                out_bytes=b * n * esize,
                modes=cfg.modes,
            )
            yield Case(
                op="linear_decode_3d",
                variant=f"B{b}",
                fn=(lambda t=x3, ww=w: F.linear(t, ww)),
                tags={"pattern": "E", "issue": 189847, "fixed_by": 189855},
                **common,
            )
            yield Case(
                op="linear_decode_2d",
                variant=f"B{b}",
                fn=(lambda t=x2, ww=w: F.linear(t, ww)),
                tags={"pattern": "E", "role": "reference for linear_decode_3d"},
                **common,
            )


def _gemv_cases(cfg: SweepConfig) -> Iterator[Case]:
    """True matrix-vector: purely bandwidth-bound, got dedicated kernels in
    #186927 (2026-07-16 -> 2.14.0)."""
    for m, k in ((4096, 4096), (11008, 4096)):
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            a = torch.empty(m, k, dtype=dtype, device="mps").uniform_(-1, 1)
            v = torch.empty(k, dtype=dtype, device="mps").uniform_(-1, 1)
            esize = a.element_size()
            yield Case(
                group="matmul",
                op="mv",
                variant=f"{m}x{k}",
                dtype=dtype_name,
                bound="memory",
                shape=(m, k),
                fn=(lambda x=a, y=v: torch.mv(x, y)),
                bytes_moved=(m * k + k + m) * esize,
                flops=2 * m * k,
                out_bytes=m * esize,
                modes=cfg.modes,
                tags={"pattern": "E", "pr": 186927},
            )


#: (label, tokens, K, N): nn.Linear as LLMs call it, [1, tokens, K] x [N, K]^T.
#: From WORKLOADS.md; the weight is what dominates decode traffic.
MODEL_LINEAR_SHAPES = (
    ("llama1b-up-prefill", 512, 2048, 8192),
    ("llama1b-up-decode", 1, 2048, 8192),
    ("llama1b-lmhead-decode", 1, 2048, 128256),
    ("gemma2-up-prefill", 512, 2304, 9216),
    ("qwen3-qkv-decode", 1, 1024, 3072),
)
QUICK_MODEL_LINEAR = ("llama1b-up-prefill", "llama1b-up-decode", "llama1b-lmhead-decode")

#: (label, batch*heads, L, D): eager attention scores q @ k^T, as the math SDPA
#: path and training backward run it (Llama-3.2-1B train, seq 256).
BMM_SHAPES = (("llama1b-train-scores", 32, 256, 64),)


def _model_linear_cases(cfg: SweepConfig) -> Iterator[Case]:
    for label, tokens, k, n in MODEL_LINEAR_SHAPES:
        if cfg.quick and label not in QUICK_MODEL_LINEAR:
            continue
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            x = torch.empty(1, tokens, k, dtype=dtype, device="mps").uniform_(-1, 1)
            w = torch.empty(n, k, dtype=dtype, device="mps").uniform_(-1, 1)
            esize = w.element_size()
            yield Case(
                group="matmul",
                op="linear_model",
                variant=label,
                dtype=dtype_name,
                bound="memory" if tokens == 1 else "compute",
                shape=(1, tokens, k, n),
                fn=(lambda t=x, ww=w: F.linear(t, ww)),
                bytes_moved=(tokens * k + n * k + tokens * n) * esize,
                flops=2 * tokens * k * n,
                out_bytes=tokens * n * esize,
                modes=cfg.modes,
                tags={"source": "WORKLOADS.md"},
            )


def _bmm_cases(cfg: SweepConfig) -> Iterator[Case]:
    for label, bh, seq, d in BMM_SHAPES:
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            # q as models produce it ([B, L, H, D] viewed to heads), k^T as a view.
            q = torch.empty(seq, bh, d, dtype=dtype, device="mps").uniform_(-1, 1).transpose(0, 1)
            k = torch.empty(bh, seq, d, dtype=dtype, device="mps").uniform_(-1, 1)
            esize = q.element_size()
            yield Case(
                group="matmul",
                op="bmm",
                variant=label,
                dtype=dtype_name,
                bound="compute",
                shape=(bh, seq, d, seq),
                fn=(lambda a=q, b=k: torch.bmm(a, b.transpose(1, 2))),
                bytes_moved=(2 * bh * seq * d + bh * seq * seq) * esize,
                flops=2 * bh * seq * seq * d,
                out_bytes=bh * seq * seq * esize,
                modes=cfg.modes,
                tags={"source": "WORKLOADS.md"},
            )


def cases(cfg: SweepConfig) -> Iterator[Case]:
    yield from _model_linear_cases(cfg)
    yield from _bmm_cases(cfg)
    yield from _decode_cases(cfg)
    yield from _gemv_cases(cfg)
    yield from _gemm_cases(cfg)
