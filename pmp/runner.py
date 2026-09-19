"""Run one sweep inside one torch environment and emit JSONL.

Invoked once per torch version by tools/sweep.py, always as a subprocess: two
torch versions cannot coexist in one interpreter, and a crash in an old wheel
must not take the orchestrator down.

Output is JSONL with two record types:
  {"type": "run",    ...}   exactly one, first line: machine, OS, peaks, torch
  {"type": "result", ...}   one per (case, mode)
Machine metadata is written once rather than stamped onto every row; the
analyzer joins on run_id.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path

import torch

from .case import DTYPES, SweepConfig
from .device import probe_machine, probe_peaks, torch_build_info
from .harness import measure
from .ops import GROUP_NAMES, cases as iter_cases


def _emit(fh, obj) -> None:
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def _derive(case, timing, peaks) -> dict:
    t = timing.median_s
    gbps = case.bytes_moved / t / 1e9 if case.bytes_moved else None
    gflops = case.flops / t / 1e9 if case.flops else None
    d = {"gbps": gbps, "gflops": gflops}
    if gbps and peaks.bandwidth_gbps:
        d["pct_peak_bw"] = 100.0 * gbps / peaks.bandwidth_gbps
    if gflops and peaks.gflops_fp32:
        ref = peaks.gflops_fp16 if case.dtype in ("float16", "bfloat16") else peaks.gflops_fp32
        d["pct_peak_flops"] = 100.0 * gflops / ref if ref else None
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pmp.runner")
    p.add_argument("--out", required=True, type=Path, help="JSONL output path")
    p.add_argument("--groups", nargs="*", default=list(GROUP_NAMES))
    p.add_argument("--dtypes", nargs="*", default=["float32", "float16", "bfloat16"])
    p.add_argument("--bounds", nargs="*", default=["launch", "memory"])
    p.add_argument("--modes", nargs="*", default=["latency", "throughput"])
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--quick", action="store_true", help="small representative subset")
    p.add_argument("--pass-index", type=int, default=0,
                   help="which interleaved pass this is; recorded for drift analysis")
    p.add_argument("--label", default="", help="free-form tag, e.g. a git sha")
    p.add_argument("--skip-peaks", action="store_true")
    args = p.parse_args(argv)

    for d in args.dtypes:
        if d not in DTYPES:
            p.error(f"unknown dtype {d!r}; known: {', '.join(DTYPES)}")

    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine", file=sys.stderr)
        return 2

    machine = probe_machine()
    tinfo = torch_build_info()
    cfg = SweepConfig(
        dtypes=tuple(args.dtypes),
        variants=tuple(args.variants) if args.variants else None,
        bounds=tuple(args.bounds),
        modes=tuple(args.modes),
        quick=args.quick,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:16]

    with args.out.open("w") as fh:
        print(f"[pmp] torch {tinfo['torch_version']} on {machine.slug}", file=sys.stderr)
        peaks = None
        if not args.skip_peaks:
            print("[pmp] probing device peaks...", file=sys.stderr)
            peaks = probe_peaks()
            print(
                f"[pmp]   bandwidth {peaks.bandwidth_gbps:.0f} GB/s  "
                f"fp32 {peaks.gflops_fp32:.0f} GFLOP/s  fp16 {peaks.gflops_fp16:.0f} GFLOP/s",
                file=sys.stderr,
            )
        else:
            from .device import Peaks

            peaks = Peaks(0.0, 0.0, 0.0, method="skipped")

        _emit(
            fh,
            {
                "type": "run",
                "schema": 1,
                "run_id": run_id,
                "ts": time.time(),
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pass_index": args.pass_index,
                "label": args.label,
                "machine": machine.to_dict(),
                "peaks": peaks.to_dict(),
                "torch": tinfo,
                "config": vars(args) | {"out": str(args.out)},
                "host_python": platform.python_version(),
            },
        )

        n_ok = n_err = 0
        t_start = time.time()
        for i, case in enumerate(iter_cases(cfg, tuple(args.groups))):
            for mode in case.modes:
                try:
                    timing = measure(case.fn, mode, out_bytes=case.out_bytes)
                except Exception:  # noqa: BLE001 - one bad op must not end the sweep
                    n_err += 1
                    _emit(
                        fh,
                        {
                            "type": "result",
                            "run_id": run_id,
                            "mode": mode,
                            "error": traceback.format_exc(limit=3),
                            **case.to_dict(),
                        },
                    )
                    continue
                n_ok += 1
                _emit(
                    fh,
                    {
                        "type": "result",
                        "run_id": run_id,
                        "mode": mode,
                        "timing": timing.to_dict(),
                        "derived": _derive(case, timing, peaks),
                        **case.to_dict(),
                    },
                )
            del case
            if i % 64 == 63:
                torch.mps.empty_cache()
                print(
                    f"[pmp]   {n_ok} measured, {n_err} failed, "
                    f"{time.time() - t_start:.0f}s elapsed",
                    file=sys.stderr,
                )

    print(
        f"[pmp] done: {n_ok} measured, {n_err} failed in "
        f"{time.time() - t_start:.0f}s -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
