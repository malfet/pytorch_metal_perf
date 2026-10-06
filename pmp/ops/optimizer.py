"""Optimizer steps: per-parameter loop vs foreach vs fused.

On MPS, torch.optim's `_default_to_fused_or_foreach` returns (False, False), so
a plain `AdamW(params)` runs the single-tensor loop: ~8 kernels per parameter
per step. WORKLOADS.md shows 1,168 launches per AdamW step on Llama-3.2-1B.
That makes this a launch-overhead group (pattern A) as much as a bandwidth one,
and the interesting ratio is default vs foreach vs fused on the same params.

Two parameter sets:
  llama1b-2layers  the real tensor shapes of two Llama-3.2-1B decoder layers
                   (~122M params): bandwidth-dominated.
  small-x300       300 tensors of 64K elements: launch-dominated, the regime
                   of many-layer small models (Qwen3-0.6B has 311 tensors).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import Case, SweepConfig

_FLOAT = ("float32", "float16", "bfloat16")

_LLAMA_LAYER = ((2048, 2048), (512, 2048), (512, 2048), (2048, 2048),
                (8192, 2048), (8192, 2048), (2048, 8192), (2048,), (2048,))

PARAM_SETS = {
    "llama1b-2layers": _LLAMA_LAYER * 2,
    "small-x300": ((256, 256),) * 300,
}

MODES = {
    "default": {},
    "foreach": {"foreach": True},
    "fused": {"fused": True},
}


def cases(cfg: SweepConfig) -> Iterator[Case]:
    sets = ["small-x300"] if cfg.quick else list(PARAM_SETS)
    for set_name in sets:
        shapes = PARAM_SETS[set_name]
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            for mode, kw in MODES.items():
                params = [torch.nn.Parameter(torch.empty(s, dtype=dtype, device="mps").uniform_(-1, 1))
                          for s in shapes]
                for p in params:
                    p.grad = torch.empty_like(p).uniform_(-1e-3, 1e-3)
                try:
                    opt = torch.optim.AdamW(params, lr=1e-6, **kw)
                    opt.step()
                except (RuntimeError, TypeError, ValueError, NotImplementedError):
                    continue
                numel = sum(p.numel() for p in params)
                esize = params[0].element_size()
                yield Case(
                    group="optimizer",
                    op="adamw",
                    variant=f"{set_name}/{mode}",
                    dtype=dtype_name,
                    bound="memory",
                    shape=(len(params), numel),
                    fn=opt.step,
                    # reads p, g, exp_avg, exp_avg_sq; writes p, exp_avg, exp_avg_sq
                    bytes_moved=7 * numel * esize,
                    out_bytes=numel * esize,
                    modes=cfg.modes,
                    tags={"pattern": "A", "source": "WORKLOADS.md", "tensors": len(params)},
                )
