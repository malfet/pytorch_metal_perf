"""Elementwise binary ops.

The canonical MPSGraph->Metal migration group, and the one with a documented
end-user regression: #152515 moved `mul` to TensorIterator for 2.8.0 and a user
bisected a 15-20% whole-application slowdown to it (#168964). The reporter's
diagnosis is worth encoding: the Metal kernel widens half inputs to fp32 for
accuracy they did not need. So fp16/bf16 lanes are never averaged with fp32.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import LAUNCH_SHAPE, MEMORY_SHAPE, Case, SweepConfig
from ..layouts import make

#: (label, variant_a, variant_b). `scalar` and `cpu_scalar` are not layouts of a
#: same-shaped operand -- they select a different dispatch path entirely.
PAIRS = (
    ("dense-dense", "dense", "dense"),
    ("transp-transp", "transposed", "transposed"),
    ("dense-transp", "dense", "transposed"),
    ("islice-islice", "inner_slice", "inner_slice"),
    ("oslice-dense", "outer_slice", "dense"),
    ("offset1-dense", "offset1", "dense"),
    ("dense-bcast", "dense", "broadcast"),
    ("dense-scalar", "dense", "scalar"),
    ("dense-cpuscalar", "dense", "cpu_scalar"),
)

QUICK_PAIRS = ("dense-dense", "transp-transp", "islice-islice", "dense-scalar")

_FLOAT = ("float32", "float16", "bfloat16")
_ALL_NUM = _FLOAT + ("int64", "int32")

#: op -> (callable, dtypes it accepts, flops per element)
OPS = {
    "add": (torch.add, _ALL_NUM, 1),
    "mul": (torch.mul, _ALL_NUM, 1),
    "div": (torch.div, _FLOAT, 1),
    "maximum": (torch.maximum, _ALL_NUM, 1),
    "pow": (torch.pow, _FLOAT, 8),
    "atan2": (torch.atan2, _FLOAT, 20),
    "lt": (torch.lt, _ALL_NUM, 1),
}

QUICK_OPS = ("add", "mul", "div", "pow")


def _operand(variant: str, shape, dtype):
    """Second operand. `scalar`/`cpu_scalar` deliberately are not device tensors."""
    if variant == "scalar":
        return 1.5 if dtype.is_floating_point else 3
    if variant == "cpu_scalar":
        # 0-dim CPU tensor: MPS has a dedicated fast path for this, and it has
        # been wrong before (#187229 ignored its storage_offset).
        return torch.tensor(1.5 if dtype.is_floating_point else 3, dtype=dtype)
    return make(shape, dtype, variant)


def cases(cfg: SweepConfig) -> Iterator[Case]:
    pairs = [p for p in PAIRS if not cfg.quick or p[0] in QUICK_PAIRS]
    op_names = [o for o in OPS if not cfg.quick or o in QUICK_OPS]
    if cfg.variants:
        pairs = [p for p in pairs if p[0] in cfg.variants]

    for op_name in op_names:
        fn, ok_dtypes, flops_per_elem = OPS[op_name]
        for dtype_name in cfg.dtypes:
            if dtype_name not in ok_dtypes:
                continue
            dtype = getattr(torch, dtype_name)
            for bound in cfg.bounds:
                shape = LAUNCH_SHAPE if bound == "launch" else MEMORY_SHAPE
                for label, va, vb in pairs:
                    try:
                        a = make(shape, dtype, va)
                        b = _operand(vb, shape, dtype)
                    except (ValueError, RuntimeError):
                        continue

                    esize = a.element_size()
                    numel = a.numel()
                    # Comparisons emit bool, everything else emits the input dtype.
                    out_esize = 1 if op_name == "lt" else esize
                    reads = numel * esize * (2 if torch.is_tensor(b) and b.numel() > 1 else 1)
                    out_bytes = numel * out_esize

                    yield Case(
                        group="binary",
                        op=op_name,
                        variant=label,
                        dtype=dtype_name,
                        bound=bound,
                        shape=shape,
                        fn=(lambda f=fn, x=a, y=b: f(x, y)),
                        bytes_moved=reads + out_bytes,
                        flops=numel * flops_per_elem,
                        out_bytes=out_bytes,
                        modes=cfg.modes,
                        tags={"pattern": "A/B/D/J", "issue": 168964}
                        if op_name == "mul"
                        else {},
                    )
