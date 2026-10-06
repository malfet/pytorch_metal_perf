"""Copies, casts and concatenation -- the data movement models do between ops.

None of these compute anything, so each should run at the device's copy
bandwidth; any shortfall is the layout or dispatch cost the backend charges.
They are 2-20% of model traffic in WORKLOADS.md and come in a few shapes:

  kv_cat        KV-cache append during decode: cat([B,H,L,D], [B,H,1,D], -2).
                Rewrites the whole cache every token.
  cast_up/down  fp32 round trips around RMSNorm in bf16 LLMs (Qwen3, Gemma 2).
  heads_contig  [B,L,H,D] -> [B,H,L,D] .contiguous(): attention head shuffles.
  t_contig      x.t().contiguous(): a plain 2D transpose.
  window_part   Swin's window partition: view + permute + contiguous.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ..case import Case, SweepConfig

_FLOAT = ("float32", "float16", "bfloat16")
_HALF = ("float16", "bfloat16")

#: (label, B, H, L, D) for the KV-cache append.
KV_SHAPES = (
    ("llama1b-512", 1, 8, 512, 64),
    ("llama8b-512", 1, 8, 512, 128),
    ("phi2-512", 1, 32, 512, 80),
    ("llama1b-4096", 1, 8, 4096, 64),
)

#: (label, B, L, H, D) for the head permute.
HEAD_SHAPES = (
    ("llama1b-prefill", 1, 512, 32, 64),
    ("vit-b16", 8, 197, 12, 64),
    ("sd-unet-4096", 2, 4096, 8, 40),
)

#: (label, rows, width) for casts.
CAST_SHAPES = (
    ("qwen3-hidden", 512, 1024),
    ("llama1b-hidden", 512, 2048),
    ("memory", 4096, 4096),
)


def _case(op, variant, dtype_name, shape, fn, nbytes, out_bytes, cfg, **tags):
    return Case(group="copy", op=op, variant=variant, dtype=dtype_name, bound="memory",
                shape=shape, fn=fn, bytes_moved=nbytes, out_bytes=out_bytes,
                modes=cfg.modes, tags={"source": "WORKLOADS.md", **tags})


def cases(cfg: SweepConfig) -> Iterator[Case]:
    for dtype_name in cfg.dtypes:
        if dtype_name not in _FLOAT:
            continue
        dtype = getattr(torch, dtype_name)
        esize = torch.empty((), dtype=dtype).element_size()

        for label, b, h, seq, d in KV_SHAPES:
            if cfg.quick and label != "llama1b-512":
                continue
            cache = torch.empty(b, h, seq, d, dtype=dtype, device="mps").uniform_()
            new = torch.empty(b, h, 1, d, dtype=dtype, device="mps").uniform_()
            out = b * h * (seq + 1) * d * esize
            yield _case("kv_cat", label, dtype_name, (b, h, seq, d),
                        lambda c=cache, n=new: torch.cat([c, n], dim=-2),
                        2 * out, out, cfg)

        for label, b, seq, h, d in HEAD_SHAPES:
            if cfg.quick and label != "vit-b16":
                continue
            x = torch.empty(b, seq, h, d, dtype=dtype, device="mps").uniform_().transpose(1, 2)
            nb = x.numel() * esize
            yield _case("heads_contig", label, dtype_name, (b, h, seq, d),
                        lambda t=x: t.contiguous(), 2 * nb, nb, cfg)

        x = torch.empty(4096, 4096, dtype=dtype, device="mps").uniform_()
        nb = x.numel() * esize
        yield _case("t_contig", "4096x4096", dtype_name, (4096, 4096),
                    lambda t=x: t.t().contiguous(), 2 * nb, nb, cfg)

        # Swin-T stage 1: [8, 56, 56, 96] into 7x7 windows.
        x = torch.empty(8, 56, 56, 96, dtype=dtype, device="mps").uniform_()
        nb = x.numel() * esize
        yield _case("window_part", "swin_t-s1", dtype_name, (8, 56, 56, 96),
                    lambda t=x: t.view(8, 8, 7, 8, 7, 96).permute(0, 1, 3, 2, 4, 5).contiguous(),
                    2 * nb, nb, cfg)

        if dtype_name in _HALF:
            for label, rows, width in CAST_SHAPES:
                if cfg.quick and label != "llama1b-hidden":
                    continue
                lo = torch.empty(rows, width, dtype=dtype, device="mps").uniform_()
                hi = torch.empty(rows, width, dtype=torch.float32, device="mps").uniform_()
                n = rows * width
                yield _case("cast_up", label, dtype_name, (rows, width),
                            lambda t=lo: t.float(), n * (esize + 4), n * 4, cfg)
                yield _case("cast_down", label, dtype_name, (rows, width),
                            lambda t=hi, dt=dtype: t.to(dt), n * (esize + 4), n * esize, cfg)
