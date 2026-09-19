"""Elementwise unary ops.

Spans the ALU-intensity range on purpose: `neg`/`abs` are pure bandwidth, so at
the memory-bound shape they should sit near the measured device peak and any
shortfall is a layout or dispatch problem. `erf`/`sin`/`exp` are transcendental,
so the same shape becomes partly ALU-bound and a precision-widening change
(pattern B) shows up there and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import LAUNCH_SHAPE, MEMORY_SHAPE, Case, SweepConfig
from ..layouts import CORE_VARIANTS, make

_FLOAT = ("float32", "float16", "bfloat16")
_ALL_NUM = _FLOAT + ("int64", "int32")

#: op -> (callable, dtypes, approximate flops per element)
OPS = {
    "neg": (torch.neg, _ALL_NUM, 1),
    "abs": (torch.abs, _ALL_NUM, 1),
    "sqrt": (torch.sqrt, _FLOAT, 4),
    "rsqrt": (torch.rsqrt, _FLOAT, 4),
    "exp": (torch.exp, _FLOAT, 10),
    "sin": (torch.sin, _FLOAT, 15),
    "erf": (torch.erf, _FLOAT, 25),
    "sigmoid": (torch.sigmoid, _FLOAT, 12),
    "tanh": (torch.tanh, _FLOAT, 15),
    "isnan": (torch.isnan, _FLOAT, 1),
}

QUICK_OPS = ("neg", "sqrt", "exp", "erf")
QUICK_VARIANTS = ("dense", "transposed", "inner_slice")


def cases(cfg: SweepConfig) -> Iterator[Case]:
    op_names = [o for o in OPS if not cfg.quick or o in QUICK_OPS]
    variants = cfg.variants or (QUICK_VARIANTS if cfg.quick else CORE_VARIANTS)

    for op_name in op_names:
        fn, ok_dtypes, flops_per_elem = OPS[op_name]
        for dtype_name in cfg.dtypes:
            if dtype_name not in ok_dtypes:
                continue
            dtype = getattr(torch, dtype_name)
            for bound in cfg.bounds:
                shape = LAUNCH_SHAPE if bound == "launch" else MEMORY_SHAPE
                for variant in variants:
                    try:
                        x = make(shape, dtype, variant)
                    except (ValueError, RuntimeError):
                        continue

                    numel = x.numel()
                    esize = x.element_size()
                    out_esize = 1 if op_name == "isnan" else esize
                    out_bytes = numel * out_esize

                    yield Case(
                        group="unary",
                        op=op_name,
                        variant=variant,
                        dtype=dtype_name,
                        bound=bound,
                        shape=shape,
                        fn=(lambda f=fn, t=x: f(t)),
                        bytes_moved=numel * esize + out_bytes,
                        flops=numel * flops_per_elem,
                        out_bytes=out_bytes,
                        modes=cfg.modes,
                        tags={"pattern": "D", "prs": [185291, 188483]}
                        if variant in ("inner_slice", "outer_slice")
                        else {},
                    )
