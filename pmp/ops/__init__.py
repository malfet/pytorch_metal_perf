"""Op group registry.

Each module exposes `cases(cfg: SweepConfig) -> Iterator[Case]`. Groups are the
aggregation unit in the report: a geomean is only meaningful over ops that share
a bottleneck, so `unary` and `matmul` never get averaged together.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from ..case import Case, SweepConfig

#: Registration order is report order.
GROUP_NAMES = (
    "unary",
    "binary",
    "reduction",
    "normalization",
    "matmul",
)


def load(name: str):
    return importlib.import_module(f"{__package__}.{name}")


def cases(cfg: SweepConfig, groups: tuple[str, ...] | None = None) -> Iterator[Case]:
    for name in groups or GROUP_NAMES:
        if name not in GROUP_NAMES:
            raise KeyError(f"unknown group {name!r}; known: {', '.join(GROUP_NAMES)}")
        yield from load(name).cases(cfg)
