"""Reductions.

The most actively rewritten MPS area in the 2.9->2.14 window: #187313 (faster
non-contiguous), then the six-part #191097-#191104 series (inner-dim, strided
outer, small-dim/narrow, argmax/argmin split-K, min/max off MPSGraph, and a
correctness fix for int64 over partial simdgroups). If the harness is sound,
2.13->2.14 should show the largest per-group win here of anywhere in the suite.

Reduction *axis* is a first-class variant, not a detail: inner-dim reductions
(dim=-1, coalesced) and outer-dim reductions (dim=0, strided) hit completely
different kernels, and historically only one of them was fast at a time.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import LAUNCH_SHAPE, MEMORY_SHAPE, Case, SweepConfig
from ..layouts import make

_FLOAT = ("float32", "float16", "bfloat16")
_ALL_NUM = _FLOAT + ("int64", "int32")

#: op -> (callable taking (tensor, dim|None), dtypes)
OPS = {
    "sum": (lambda t, d: torch.sum(t) if d is None else torch.sum(t, dim=d), _ALL_NUM),
    "mean": (lambda t, d: torch.mean(t) if d is None else torch.mean(t, dim=d), _FLOAT),
    "max": (lambda t, d: torch.max(t) if d is None else torch.max(t, dim=d), _ALL_NUM),
    "argmax": (
        lambda t, d: torch.argmax(t) if d is None else torch.argmax(t, dim=d),
        _ALL_NUM,
    ),
    "amin": (lambda t, d: torch.amin(t) if d is None else torch.amin(t, dim=d), _ALL_NUM),
    "norm": (lambda t, d: torch.linalg.norm(t) if d is None else torch.linalg.norm(t, dim=d), _FLOAT),
}
QUICK_OPS = ("sum", "max", "argmax")

#: (label, dim). None = full reduction to a scalar.
AXES = (("outer", 0), ("inner", -1), ("full", None))

VARIANTS = ("dense", "transposed", "inner_slice", "outer_slice")
QUICK_VARIANTS = ("dense", "inner_slice")

#: Narrow/wide shapes exist because "small dim" got its own kernel in #191098:
#: a 4096x8 reduction and an 8x4096 one are different problems.
EXTRA_SHAPES = (
    ("narrow", (1 << 20, 8)),
    ("wide", (8, 1 << 20)),
)


def _shapes(cfg: SweepConfig):
    for bound in cfg.bounds:
        yield bound, (LAUNCH_SHAPE if bound == "launch" else MEMORY_SHAPE), None
    if not cfg.quick and "memory" in cfg.bounds:
        for label, shape in EXTRA_SHAPES:
            yield "memory", shape, label


def cases(cfg: SweepConfig) -> Iterator[Case]:
    op_names = [o for o in OPS if not cfg.quick or o in QUICK_OPS]
    variants = cfg.variants or (QUICK_VARIANTS if cfg.quick else VARIANTS)

    for op_name in op_names:
        fn, ok_dtypes = OPS[op_name]
        for dtype_name in cfg.dtypes:
            if dtype_name not in ok_dtypes:
                continue
            dtype = getattr(torch, dtype_name)
            for bound, shape, shape_label in _shapes(cfg):
                for variant in variants:
                    # Narrow/wide probes are about the axis, not the layout.
                    if shape_label and variant != "dense":
                        continue
                    try:
                        x = make(shape, dtype, variant)
                    except (ValueError, RuntimeError):
                        continue
                    for axis_label, dim in AXES:
                        try:
                            out = fn(x, dim)
                        except (RuntimeError, IndexError):
                            continue
                        out_t = out[0] if isinstance(out, tuple) else out
                        out_bytes = out_t.numel() * out_t.element_size()

                        label = f"{variant}/{axis_label}"
                        if shape_label:
                            label = f"{shape_label}/{axis_label}"

                        yield Case(
                            group="reduction",
                            op=op_name,
                            variant=label,
                            dtype=dtype_name,
                            bound=bound,
                            shape=tuple(shape),
                            fn=(lambda f=fn, t=x, d=dim: f(t, d)),
                            bytes_moved=x.numel() * x.element_size() + out_bytes,
                            flops=x.numel(),
                            out_bytes=out_bytes,
                            modes=cfg.modes,
                            tags={
                                "pattern": "D",
                                "prs": [187313, 191097, 191098, 191099, 191100, 191101],
                            },
                        )
