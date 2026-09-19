"""The unit of measurement: one (op, layout, dtype, shape, bound) cell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

#: Nominal sizes for the two extremes every elementwise-ish op is measured at.
#:
#: launch  -- small enough that the kernel finishes before the dispatch does, so
#:            per-encoder cost dominates. This is where a migration from one
#:            cached MPSGraph to N individually-encoded Metal kernels shows up.
#: memory  -- far past any cache, so DRAM bandwidth dominates and the result is
#:            comparable against the measured device peak.
LAUNCH_SHAPE = (256, 256)      # 256 KiB fp32
MEMORY_SHAPE = (4096, 4096)    # 64 MiB fp32

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
    "complex64": torch.complex64,
}


@dataclass
class SweepConfig:
    """What slice of the full cross product to run."""

    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16")
    variants: tuple[str, ...] | None = None   # None -> each group's default
    bounds: tuple[str, ...] = ("launch", "memory")
    modes: tuple[str, ...] = ("latency", "throughput")
    quick: bool = False

    def torch_dtypes(self) -> list[torch.dtype]:
        return [DTYPES[d] for d in self.dtypes]


@dataclass
class Case:
    """One measurable cell.

    `fn` is a zero-argument closure over already-allocated tensors: allocation
    and layout construction must happen when the Case is built, never inside the
    timed region.
    """

    group: str
    op: str
    variant: str
    dtype: str
    bound: str
    shape: tuple[int, ...]
    fn: Callable[[], object]

    #: Bytes an ideal kernel touches per call -> achieved GB/s.
    bytes_moved: int = 0
    #: FLOPs per call -> achieved GFLOP/s. Zero for memory-bound ops.
    flops: int = 0
    #: Bytes the op allocates per call; bounds in-flight work in throughput mode.
    out_bytes: int = 0

    modes: tuple[str, ...] = ("latency", "throughput")
    #: Free-form provenance, e.g. {"issue": 189847, "pattern": "E"}.
    tags: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        shape = "x".join(str(d) for d in self.shape)
        return f"{self.group}/{self.op}/{self.variant}/{self.dtype}/{shape}/{self.bound}"

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "op": self.op,
            "variant": self.variant,
            "dtype": self.dtype,
            "bound": self.bound,
            "shape": list(self.shape),
            "key": self.key,
            "bytes_moved": self.bytes_moved,
            "flops": self.flops,
            "tags": self.tags,
        }
