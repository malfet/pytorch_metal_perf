"""Convolution, at the shapes torchvision, SD 1.5 and the speech encoders run.

Every shape below is lifted from WORKLOADS.md (tools/trace_models.py), labelled
with the model it came from. They span the regimes the conv kernels split on:
1x1 (a GEMM in disguise), dense 3x3, depthwise 3x3/7x7 (groups == channels,
bandwidth-bound), large-stride patchify stems, and 1D speech front-ends.

channels_last is swept for every 2D shape because it is how perf-minded
inference runs (ConvNeXt does it internally) and because the NHWC path has
regressed independently of NCHW before (pattern F, #174241).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..case import Case, SweepConfig

_FLOAT = ("float32", "float16", "bfloat16")

#: (label, N, C_in, H, W, C_out, kernel, stride, padding, groups)
CONV2D_SHAPES = (
    ("resnet50-1x1", 8, 64, 56, 56, 256, 1, 1, 0, 1),
    ("resnet50-3x3", 8, 64, 56, 56, 64, 3, 1, 1, 1),
    ("resnet50-stem7x7", 8, 3, 224, 224, 64, 7, 2, 3, 1),
    ("mbv3-dw3x3-s2", 8, 64, 112, 112, 64, 3, 2, 1, 64),
    ("convnext-dw7x7", 8, 96, 56, 56, 96, 7, 1, 3, 96),
    ("effnet-1x1-expand", 8, 16, 112, 112, 96, 1, 1, 0, 1),
    ("vit-patch16", 8, 3, 224, 224, 768, 16, 16, 0, 1),
    ("sd-unet-3x3-1280", 2, 1280, 16, 16, 1280, 3, 1, 1, 1),
    ("sd-unet-3x3-320", 2, 320, 64, 64, 320, 3, 1, 1, 1),
    ("sd-vae-3x3-128@512", 1, 128, 512, 512, 128, 3, 1, 1, 1),
)
QUICK_2D = ("resnet50-3x3", "mbv3-dw3x3-s2", "sd-unet-3x3-320")

#: (label, N, C_in, L, C_out, kernel, stride, padding)
CONV1D_SHAPES = (
    ("wav2vec2-stem-k10s5", 1, 1, 160000, 512, 10, 5, 0),
    ("wav2vec2-k3s2", 1, 512, 31999, 512, 3, 2, 0),
    ("whisper-k3s2", 1, 768, 3000, 768, 3, 2, 1),
)
QUICK_1D = ("wav2vec2-k3s2",)


def _out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


def cases(cfg: SweepConfig) -> Iterator[Case]:
    for label, n, cin, h, w, cout, k, s, p, g in CONV2D_SHAPES:
        if cfg.quick and label not in QUICK_2D:
            continue
        ho, wo = _out(h, k, s, p), _out(w, k, s, p)
        flops = 2 * n * cout * ho * wo * (cin // g) * k * k
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            esize = torch.empty((), dtype=dtype).element_size()
            for variant in ("dense", "channels_last"):
                mf = torch.channels_last if variant == "channels_last" else torch.contiguous_format
                x = torch.empty(n, cin, h, w, dtype=dtype, device="mps").uniform_(-1, 1)
                x = x.to(memory_format=mf)
                wt = torch.empty(cout, cin // g, k, k, dtype=dtype, device="mps").uniform_(-0.1, 0.1)
                wt = wt.to(memory_format=mf)
                b = torch.zeros(cout, dtype=dtype, device="mps")
                fn = (lambda x=x, wt=wt, b=b, s=s, p=p, g=g:  # noqa: E731
                      F.conv2d(x, wt, b, stride=s, padding=p, groups=g))
                try:
                    fn()
                except (RuntimeError, NotImplementedError):
                    continue
                out_bytes = n * cout * ho * wo * esize
                yield Case(
                    group="conv",
                    op="conv2d" if g == 1 else "conv2d_dw",
                    variant=f"{label}/{variant}",
                    dtype=dtype_name,
                    bound="compute",
                    shape=(n, cin, h, w, cout, k),
                    fn=fn,
                    bytes_moved=x.numel() * esize + wt.numel() * esize + out_bytes,
                    flops=flops,
                    out_bytes=out_bytes,
                    modes=cfg.modes,
                    tags={"pattern": "F" if variant == "channels_last" else "",
                          "source": "WORKLOADS.md"},
                )

    for label, n, cin, length, cout, k, s, p in CONV1D_SHAPES:
        if cfg.quick and label not in QUICK_1D:
            continue
        lo = _out(length, k, s, p)
        for dtype_name in cfg.dtypes:
            if dtype_name not in _FLOAT:
                continue
            dtype = getattr(torch, dtype_name)
            x = torch.empty(n, cin, length, dtype=dtype, device="mps").uniform_(-1, 1)
            wt = torch.empty(cout, cin, k, dtype=dtype, device="mps").uniform_(-0.1, 0.1)
            fn = lambda x=x, wt=wt, s=s, p=p: F.conv1d(x, wt, stride=s, padding=p)  # noqa: E731
            try:
                fn()
            except (RuntimeError, NotImplementedError):
                continue
            esize = x.element_size()
            out_bytes = n * cout * lo * esize
            yield Case(
                group="conv",
                op="conv1d",
                variant=label,
                dtype=dtype_name,
                bound="compute",
                shape=(n, cin, length, cout, k),
                fn=fn,
                bytes_moved=(x.numel() + wt.numel()) * esize + out_bytes,
                flops=2 * n * cout * lo * cin * k,
                out_bytes=out_bytes,
                modes=cfg.modes,
                tags={"source": "WORKLOADS.md"},
            )
