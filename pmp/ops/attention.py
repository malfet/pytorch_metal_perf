"""scaled_dot_product_attention at the shapes real models call it with.

Shapes come from WORKLOADS.md. Two layouts per shape:

  dense    [B, H, L, D] contiguous -- what a micro-benchmark naturally builds.
  heads_T  [B, L, H, D] storage viewed as [B, H, L, D] -- what every traced
           model actually passes, since q/k/v come out of a linear projection
           as [B, L, H*D] and are only *viewed* into heads.

The gap between the two is the cost of the layout the backend is really given.
Grouped-query shapes (Llama) use enable_gqa where the torch version has it and
are skipped otherwise rather than silently measuring a repeat_interleave.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..case import Case, SweepConfig

_FLOAT = ("float32", "float16", "bfloat16")

#: (label, B, H_q, H_kv, L_q, L_kv, D, causal)
SHAPES = (
    ("llama1b-prefill-512", 1, 32, 8, 512, 512, 64, True),
    ("llama1b-decode-513", 1, 32, 8, 1, 513, 64, False),
    ("llama8b-prefill-512", 1, 32, 8, 512, 512, 128, True),
    ("bert-base-128", 8, 12, 12, 128, 128, 64, False),
    ("vit-b16-197", 8, 12, 12, 197, 197, 64, False),
    ("whisper-enc-1500", 1, 12, 12, 1500, 1500, 64, False),
    ("sd-unet-self-4096", 2, 8, 8, 4096, 4096, 40, False),
    ("sd-unet-cross-77", 2, 8, 8, 4096, 77, 40, False),
)
QUICK = ("llama1b-prefill-512", "llama1b-decode-513", "vit-b16-197", "sd-unet-self-4096")


def _make(b, h, seq, d, dtype, heads_t):
    if heads_t:
        return torch.empty(b, seq, h, d, dtype=dtype, device="mps").uniform_(-1, 1).transpose(1, 2)
    return torch.empty(b, h, seq, d, dtype=dtype, device="mps").uniform_(-1, 1)


def cases(cfg: SweepConfig) -> Iterator[Case]:
    for label, b, hq, hkv, lq, lk, d, causal in SHAPES:
        if cfg.quick and label not in QUICK:
            continue
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            for variant in ("dense", "heads_T"):
                heads_t = variant == "heads_T"
                q = _make(b, hq, lq, d, dtype, heads_t)
                k = _make(b, hkv, lk, d, dtype, heads_t)
                v = _make(b, hkv, lk, d, dtype, heads_t)
                kw = {"is_causal": causal}
                if hq != hkv:
                    kw["enable_gqa"] = True
                fn = lambda q=q, k=k, v=v, kw=kw: F.scaled_dot_product_attention(q, k, v, **kw)  # noqa: E731
                try:
                    fn()
                except (RuntimeError, TypeError, NotImplementedError):
                    continue
                esize = q.element_size()
                out_bytes = b * hq * lq * d * esize
                flops = 4 * b * hq * lq * lk * d // (2 if causal else 1)
                yield Case(
                    group="attention",
                    op="sdpa",
                    variant=f"{label}/{variant}",
                    dtype=dtype_name,
                    bound="compute",
                    shape=(b, hq, lq, lk, d),
                    fn=fn,
                    bytes_moved=(q.numel() + k.numel() + v.numel()) * esize + out_bytes,
                    flops=flops,
                    # The math path materialises the score matrix; budget for it
                    # so throughput mode does not queue gigabytes of temporaries.
                    out_bytes=out_bytes + b * hq * lq * lk * 4,
                    modes=cfg.modes,
                    tags={"source": "WORKLOADS.md", "causal": causal, "gqa": hq != hkv},
                )
