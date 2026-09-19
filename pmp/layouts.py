"""Tensor layout variants -- the axis that catches lost fast paths.

Every variant produces the *same logical shape*, so times are directly
comparable and the ratio dense:variant is the penalty the backend pays for
non-ideal memory. Historically that ratio is where MPSGraph->Metal migrations
went wrong: the dense path got a vectorized kernel and everything else fell
back to a scalar gather, which is what #185291 (2D dispatch for strided unary),
#188483 (vectorize inner-contiguous views) and #187313 (faster non-contiguous
reductions) each had to claw back.
"""

from __future__ import annotations

import torch
from torch.testing import make_tensor as _make

#: Variants applicable to any rank-2+ tensor.
CORE_VARIANTS = (
    "dense",
    "transposed",
    "inner_slice",
    "outer_slice",
    "offset1",
)

#: Adds the broadcast lane; only meaningful for binary ops.
BINARY_VARIANTS = CORE_VARIANTS + ("broadcast",)

#: 4D-only.
IMAGE_VARIANTS = ("dense", "channels_last")

_VARIANT_DOC = {
    "dense": "contiguous, the happy path",
    "transposed": "last two dims swapped; outer stride 1",
    "inner_slice": "x[..., ::2]; innermost dim non-contiguous",
    "outer_slice": "x[::2]; rows contiguous but tensor non-dense (equal-strided)",
    "offset1": "contiguous but storage_offset=1; exercises alignment assumptions",
    "broadcast": "leading dim expanded from 1; stride 0",
    "channels_last": "NCHW logical, NHWC physical",
}


def variant_doc(variant: str) -> str:
    return _VARIANT_DOC.get(variant, "")


def _fill(shape, dtype, device):
    # make_tensor handles bool/int/float/complex uniformly and exists unchanged
    # across every torch version in the matrix.
    if dtype in (torch.float16, torch.bfloat16):
        # Keep magnitudes small so fp16 reductions do not saturate to inf and
        # change the kernel's branch behaviour.
        return _make(shape, dtype=dtype, device=device, low=-2.0, high=2.0)
    return _make(shape, dtype=dtype, device=device)


def make(shape, dtype, variant: str = "dense", device: str = "mps") -> torch.Tensor:
    """Build a tensor with logical `shape` laid out according to `variant`."""
    shape = tuple(shape)

    if variant == "dense":
        return _fill(shape, dtype, device)

    if variant == "transposed":
        if len(shape) < 2:
            raise ValueError("transposed needs rank >= 2")
        swapped = shape[:-2] + (shape[-1], shape[-2])
        return _fill(swapped, dtype, device).transpose(-1, -2)

    if variant == "inner_slice":
        wide = shape[:-1] + (shape[-1] * 2,)
        return _fill(wide, dtype, device)[..., ::2]

    if variant == "outer_slice":
        tall = (shape[0] * 2,) + shape[1:]
        return _fill(tall, dtype, device)[::2]

    if variant == "offset1":
        n = 1
        for d in shape:
            n *= d
        flat = _fill((n + 1,), dtype, device)
        return flat.narrow(0, 1, n).view(shape)

    if variant == "broadcast":
        if len(shape) < 2:
            raise ValueError("broadcast needs rank >= 2")
        return _fill((1,) + shape[1:], dtype, device).expand(shape)

    if variant == "channels_last":
        if len(shape) != 4:
            raise ValueError("channels_last needs rank 4")
        return _fill(shape, dtype, device).to(memory_format=torch.channels_last)

    raise ValueError(f"unknown layout variant: {variant!r}")


def describe(t: torch.Tensor) -> dict:
    """Layout facts worth carrying into the result record.

    `is_dense` (all storage covered, any order) is separate from `is_contiguous`
    because MPS treats them differently: a dense-but-permuted tensor can often
    take a fast path that a genuinely strided view cannot.
    """
    numel = t.numel()
    try:
        dense = bool(t.is_contiguous() or t.transpose(-1, -2).is_contiguous())
    except (RuntimeError, IndexError):
        dense = bool(t.is_contiguous())
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": t.storage_offset(),
        "is_contiguous": bool(t.is_contiguous()),
        "is_dense": dense,
        "numel": numel,
        "nbytes": numel * t.element_size(),
    }


def logical_bytes(*tensors: torch.Tensor) -> int:
    """Bytes an ideal kernel would touch: one read per input element.

    Charged on *logical* elements even for strided views, so that GB/s directly
    exposes the layout penalty instead of hiding it in a different denominator.
    """
    return sum(t.numel() * t.element_size() for t in tensors)
