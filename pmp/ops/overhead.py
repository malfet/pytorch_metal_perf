"""Floor probes: what an op costs when it does essentially no work.

Without these, the `launch`-bound lane is uninterpretable. A 256x256 elementwise
op might take 30us because the kernel is slow, or because 30us is simply what it
costs to get from Python through the dispatcher into a command encoder and back.
Those call for completely different responses, and the only way to tell them
apart is to measure the floor directly.

Interpretation:
  - a launch-bound cell at roughly the floor is dispatch-bound; its kernel is
    irrelevant and optimising it will change nothing
  - the floor itself moving between torch versions is a dispatcher or
    encoder-setup change, which is exactly what pattern A predicts when an op
    stops being one cached MPSGraph and becomes N Metal encoders

`view` is included as the pure-Python floor: it allocates no output and encodes
no GPU work, so it isolates interpreter plus dispatcher cost from everything
Metal does.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import Case, SweepConfig


def cases(cfg: SweepConfig) -> Iterator[Case]:
    dtype_name = "float32" if "float32" in cfg.dtypes else cfg.dtypes[0]
    dtype = getattr(torch, dtype_name)

    tiny = torch.ones(1, 1, dtype=dtype, device="mps")
    small = torch.ones(32, 32, dtype=dtype, device="mps")
    esize = tiny.element_size()

    probes = (
        # label, callable, out_bytes, what it isolates
        ("view", lambda t=tiny: t.view(1, 1), 0),
        ("empty_like", lambda t=tiny: torch.empty_like(t), esize),
        ("neg_1elem", lambda t=tiny: torch.neg(t), esize),
        ("add_1elem", lambda t=tiny: torch.add(t, t), esize),
        ("neg_32x32", lambda t=small: torch.neg(t), 32 * 32 * esize),
        ("mm_1x1", lambda t=tiny: t @ t, esize),
        ("sum_1elem", lambda t=tiny: torch.sum(t), esize),
    )

    for label, fn, out_bytes in probes:
        yield Case(
            group="overhead",
            op=label,
            variant="floor",
            dtype=dtype_name,
            bound="launch",
            shape=(1, 1),
            fn=fn,
            bytes_moved=0,      # GB/s is meaningless here; ms is the whole point
            flops=0,
            out_bytes=out_bytes,
            modes=cfg.modes,
            tags={"pattern": "A", "role": "dispatch floor"},
        )
