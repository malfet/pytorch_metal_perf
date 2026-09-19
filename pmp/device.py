"""Machine and OS fingerprinting, plus measured device peaks.

The OS build is part of the *identity* of a result series, not decoration.
macOS releases change which kernel path MPS takes -- MetalPerformancePrimitives
availability is OS-gated, MPSGraph rewrites its own plans between releases, and
PyTorch itself carries `is_macos_or_newer` branches. Comparing a torch 2.9 run
captured on macOS 15 against a torch 2.14 run captured on macOS 26 measures the
OS upgrade at least as much as it measures PyTorch. So every record carries an
`env_id` derived from (chip, OS build, framework versions), and the analyzer
refuses to put two different env_ids on the same axis.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import plistlib
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

_FRAMEWORKS = (
    "/System/Library/Frameworks/Metal.framework",
    "/System/Library/Frameworks/MetalPerformanceShaders.framework",
    "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework",
    "/System/Library/Frameworks/MetalPerformancePrimitives.framework",
)


def _sh(*args: str) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _framework_version(path: str) -> str | None:
    """CFBundleVersion of a system framework, or None if it is not installed.

    Absence is meaningful: MetalPerformancePrimitives only exists on recent
    macOS, and its presence flips PyTorch onto entirely different matmul and
    attention kernels.
    """
    plist = Path(path) / "Resources" / "Info.plist"
    if not plist.exists():
        plist = Path(path) / "Versions" / "Current" / "Resources" / "Info.plist"
    if not plist.exists():
        return None
    try:
        with plist.open("rb") as fh:
            info = plistlib.load(fh)
    except (OSError, ValueError):
        return "present"
    return str(info.get("CFBundleVersion") or info.get("CFBundleShortVersionString") or "present")


def _gpu_info() -> tuple[str, int | None]:
    """(GPU name, core count) from system_profiler. Slow (~1s), so call once."""
    raw = _sh("system_profiler", "-json", "SPDisplaysDataType")
    if not raw:
        return ("", None)
    try:
        items = json.loads(raw).get("SPDisplaysDataType", [])
    except json.JSONDecodeError:
        return ("", None)
    if not items:
        return ("", None)
    gpu = items[0]
    name = gpu.get("sppci_model", "")
    cores = gpu.get("sppci_cores")
    try:
        cores = int(cores) if cores is not None else None
    except (TypeError, ValueError):
        cores = None
    return (name, cores)


@dataclass
class Machine:
    """Everything that must match for two runs to be comparable."""

    chip: str
    arch: str
    cpu_cores: int | None
    gpu_name: str
    gpu_cores: int | None
    memory_gb: float | None
    os_product: str          # e.g. "macOS"
    os_version: str          # e.g. "26.6.2"
    os_build: str            # e.g. "25G83"   <- the one that actually matters
    darwin: str              # uname -r
    frameworks: dict[str, str | None] = field(default_factory=dict)

    @property
    def env_id(self) -> str:
        """Short stable hash over everything that changes kernel selection.

        Deliberately excludes torch version: the whole point is to vary torch
        while holding this fixed.
        """
        payload = json.dumps(
            {
                "chip": self.chip,
                "gpu_name": self.gpu_name,
                "gpu_cores": self.gpu_cores,
                "os_build": self.os_build,
                "frameworks": self.frameworks,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def slug(self) -> str:
        """Human-readable directory name: 'M4-Pro-20c_macOS-26.6.2-25G83'."""
        chip = re.sub(r"^Apple\s+", "", self.chip).replace(" ", "-")
        cores = f"-{self.gpu_cores}c" if self.gpu_cores else ""
        return f"{chip}{cores}_macOS-{self.os_version}-{self.os_build}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["env_id"] = self.env_id
        d["slug"] = self.slug
        return d


def probe_machine() -> Machine:
    gpu_name, gpu_cores = _gpu_info()
    mem = _sh("sysctl", "-n", "hw.memsize")
    try:
        cpu_cores = int(_sh("sysctl", "-n", "hw.ncpu") or 0) or None
    except ValueError:
        cpu_cores = None
    return Machine(
        chip=_sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        arch=platform.machine(),
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
        gpu_cores=gpu_cores,
        memory_gb=round(int(mem) / 2**30, 1) if mem.isdigit() else None,
        os_product=_sh("sw_vers", "-productName") or "macOS",
        os_version=_sh("sw_vers", "-productVersion"),
        os_build=_sh("sw_vers", "-buildVersion"),
        darwin=platform.release(),
        frameworks={
            Path(p).stem: _framework_version(p) for p in _FRAMEWORKS
        },
    )


# --------------------------------------------------------------------------
# Measured peaks
# --------------------------------------------------------------------------
# Deliberately measured with plain torch ops rather than a bespoke Metal kernel:
# this code has to run unmodified on every torch version in the matrix, and
# `torch.mps._compile_shader` is neither present nor stable across 2.9..2.15.
# What we want for normalization is *achievable* throughput anyway -- the number
# a well-written PyTorch op could reach -- not the marketing peak.


@dataclass
class Peaks:
    bandwidth_gbps: float
    gflops_fp32: float
    gflops_fp16: float
    method: str = "torch-ops"

    def to_dict(self) -> dict:
        return asdict(self)


def probe_peaks(torch_mod=None) -> Peaks:
    import torch as _t

    torch = torch_mod or _t
    from .harness import time_throughput

    dev = "mps"

    # Bandwidth: a straight copy of a buffer far larger than any cache.
    n = 1 << 25  # 32M elements = 128 MiB fp32
    src = torch.empty(n, dtype=torch.float32, device=dev).uniform_()
    dst = torch.empty_like(src)
    nbytes = src.numel() * src.element_size()
    r = time_throughput(lambda: dst.copy_(src), out_bytes=0, min_iters=8)
    # copy_ reads nbytes and writes nbytes
    bw = 2 * nbytes / r.median_s / 1e9
    del src, dst

    # FLOPs: a square matmul big enough to be firmly compute-bound.
    m = 2048
    flops = 2 * m**3
    gf = {}
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        a = torch.empty(m, m, dtype=dt, device=dev).uniform_()
        b = torch.empty(m, m, dtype=dt, device=dev).uniform_()
        r = time_throughput(lambda: a @ b, out_bytes=m * m * a.element_size(), min_iters=8)
        gf[name] = flops / r.median_s / 1e9
        del a, b

    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return Peaks(bandwidth_gbps=bw, gflops_fp32=gf["fp32"], gflops_fp16=gf["fp16"])


def torch_build_info(torch_mod=None) -> dict:
    import torch as _t

    torch = torch_mod or _t
    info = {
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    # Optional across versions -- guard every one.
    rec = getattr(torch.mps, "recommended_max_memory", None)
    if callable(rec):
        try:
            info["mps_recommended_max_gb"] = round(rec() / 2**30, 1)
        except Exception:  # noqa: BLE001 - purely informational
            pass
    for attr in ("is_macos13_or_newer", "is_macos_or_newer"):
        fn = getattr(torch.backends.mps, attr, None)
        if callable(fn):
            info["mps_gate_fn"] = attr
            break
    info["env"] = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("PYTORCH_MPS", "PYTORCH_ENABLE_MPS", "TORCHINDUCTOR", "MTL_"))
    }
    return info
