"""Normalization and softmax.

Two-pass statistics over a reduction axis followed by an elementwise rescale, so
these sit between the `reduction` and `unary` groups and inherit both groups'
failure modes: they are sensitive to the reduction-axis kernel selection *and*
to precision widening, because the mean/variance accumulation dtype is a free
choice the backend keeps revisiting.

Relevant history in the window:
  #190492  native_layer_norm needed a two-pass kernel for small-variance rows.
  #190055  mixed-dtype affine params in layer_norm forward and backward.
  #189617  fused RMSNorm had to do the weight multiply in fp32 to match CPU.

`normalized_shape` is swept across the last dim because the kernel choice keys
off how wide the reduction is: a 128-wide norm (a small transformer) and a
16384-wide one (an LLM hidden size) are different problems.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..case import Case, SweepConfig

_FLOAT = ("float32", "float16", "bfloat16")

#: (label, rows, normalized_width). Widths straddle simdgroup (32) and
#: threadgroup multiples, where the narrow/wide kernel split lives.
SHAPES = (
    ("bert", 4096, 768),
    ("llm-hidden", 4096, 4096),
    ("wide-16k", 512, 16384),
    ("narrow-128", 16384, 128),
    ("odd-1023", 4096, 1023),
)
QUICK_SHAPES = ("bert", "wide-16k", "narrow-128")


def _rms_norm(x, weight):
    # torch.rms_norm is not available across the whole matrix; the explicit form
    # is what most model code actually runs anyway.
    fn = getattr(F, "rms_norm", None)
    if fn is not None:
        return fn(x, (x.shape[-1],), weight)
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + 1e-6).to(x.dtype)) * weight


def cases(cfg: SweepConfig) -> Iterator[Case]:
    shapes = [s for s in SHAPES if not cfg.quick or s[0] in QUICK_SHAPES]
    for label, rows, width in shapes:
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            x = torch.empty(rows, width, dtype=dtype, device="mps").uniform_(-2, 2)
            w = torch.empty(width, dtype=dtype, device="mps").uniform_(0.5, 1.5)
            b = torch.zeros(width, dtype=dtype, device="mps")
            esize = x.element_size()
            nbytes = 2 * x.numel() * esize   # one read pass, one write pass
            out_bytes = x.numel() * esize

            variants = {
                "layer_norm": lambda t=x, ww=w, bb=b, n=width: F.layer_norm(t, (n,), ww, bb),
                "layer_norm_noaffine": lambda t=x, n=width: F.layer_norm(t, (n,)),
                "rms_norm": lambda t=x, ww=w: _rms_norm(t, ww),
                "softmax": lambda t=x: F.softmax(t, dim=-1),
                "log_softmax": lambda t=x: F.log_softmax(t, dim=-1),
                # dim=0 crosses rows: strided access, a different kernel entirely.
                "softmax_dim0": lambda t=x: F.softmax(t, dim=0),
            }
            for op_name, fn in variants.items():
                try:
                    fn()
                except (RuntimeError, TypeError):
                    continue
                yield Case(
                    group="normalization",
                    op=op_name,
                    variant=label,
                    dtype=dtype_name,
                    bound="memory",
                    shape=(rows, width),
                    fn=fn,
                    bytes_moved=nbytes,
                    flops=x.numel() * 8,
                    out_bytes=out_bytes,
                    modes=cfg.modes,
                    tags={"pattern": "B/D", "prs": [190492, 190055, 189617]},
                )
