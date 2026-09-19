#!/usr/bin/env python3
"""Orchestrate the sweep across torch versions.

Two things this does that matter more than they look:

1. *Interleaves* passes (v1,v2,v3, v1,v2,v3, ...) rather than running each
   version to completion. A laptop under sustained GPU load throttles, so a
   version-at-a-time schedule systematically penalises whichever version runs
   last. Interleaving turns thermal drift into noise shared by all versions
   instead of a bias against one.

2. Refuses to mix OS builds. macOS gates which kernel MPS selects, so a series
   spanning an OS upgrade is not a torch comparison. Results are filed under the
   machine+OS slug and the analyzer treats a new slug as a new series.

Each version runs as a subprocess against its own uv venv; `pmp` itself is
supplied via PYTHONPATH rather than installed, so the venvs hold nothing but
torch and its dependencies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENVS = REPO / ".envs"
DEFAULT_VERSIONS = ("2.9.0", "2.11.0", "2.13.0", "2.14.0", "local")
PYTHON = "3.13"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw)


def env_python(version: str) -> Path:
    """Interpreter for `version`, creating and populating its venv on demand."""
    if version == "local":
        return Path(sys.executable)
    venv = ENVS / f"torch-{version}"
    py = venv / "bin" / "python"
    if py.exists():
        return py
    print(f"[sweep] creating venv for torch {version}", file=sys.stderr)
    ENVS.mkdir(exist_ok=True)
    _run(["uv", "venv", "--python", PYTHON, str(venv)])
    _run(["uv", "pip", "install", "--python", str(py), f"torch=={version}", "numpy"])
    return py


def installed_version(py: Path) -> str:
    out = subprocess.run(
        [str(py), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def machine_slug(py: Path) -> str:
    out = subprocess.run(
        [str(py), "-c", "from pmp.device import probe_machine; print(probe_machine().slug)"],
        capture_output=True, text=True, check=True,
        env=os.environ | {"PYTHONPATH": str(REPO)},
    )
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sweep")
    p.add_argument("--versions", nargs="*", default=list(DEFAULT_VERSIONS))
    p.add_argument("--passes", type=int, default=3,
                   help="interleaved repetitions; median across passes is the reported number")
    p.add_argument("--groups", nargs="*", default=None)
    p.add_argument("--dtypes", nargs="*", default=None)
    p.add_argument("--bounds", nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cooldown", type=float, default=10.0,
                   help="seconds idle between runs so the GPU sheds heat (0 to disable)")
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--setup-only", action="store_true", help="build venvs and exit")
    args = p.parse_args(argv)

    pythons = {v: env_python(v) for v in args.versions}
    if args.setup_only:
        for v, py in pythons.items():
            print(f"{v:>8} -> {installed_version(py)}  ({py})")
        return 0

    slug = machine_slug(Path(sys.executable))
    print(f"[sweep] machine/OS series: {slug}", file=sys.stderr)
    print(f"[sweep] {len(args.versions)} versions x {args.passes} interleaved passes",
          file=sys.stderr)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    child_env = os.environ | {"PYTHONPATH": str(REPO)}
    failures: list[tuple[str, int]] = []

    for pass_i in range(args.passes):
        # Rotate the starting version each pass. Interleaving alone is not
        # enough: with a fixed order, the first version always runs on a cool
        # GPU and the last always runs hot, so the thermal penalty lands on the
        # same version every pass instead of averaging out. Measured drift over
        # one five-version pass was ~5% of bandwidth and ~13% of fp32 FLOP/s.
        order = args.versions[pass_i % len(args.versions):] + \
            args.versions[:pass_i % len(args.versions)]
        for version in order:
            if args.cooldown:
                time.sleep(args.cooldown)
            py = pythons[version]
            actual = installed_version(py)
            out = args.results / slug / actual / f"{stamp}-p{pass_i}.jsonl"
            print(f"[sweep] pass {pass_i}: torch {actual}", file=sys.stderr)
            cmd = [
                str(py), "-m", "pmp.runner",
                "--out", str(out),
                "--pass-index", str(pass_i),
                "--label", version,
            ]
            for flag, val in (("--groups", args.groups), ("--dtypes", args.dtypes),
                              ("--bounds", args.bounds)):
                if val:
                    cmd += [flag, *val]
            if args.quick:
                cmd.append("--quick")
            rc = subprocess.run(cmd, env=child_env, check=False).returncode
            if rc != 0:
                print(f"[sweep] !! torch {actual} pass {pass_i} exited {rc}", file=sys.stderr)
                failures.append((actual, pass_i))

    print(f"[sweep] complete -> {args.results / slug}", file=sys.stderr)
    if failures:
        print(f"[sweep] {len(failures)} failed run(s): {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
