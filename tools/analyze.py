#!/usr/bin/env python3
"""Turn sweep JSONL into per-group trends and a regression list.

Deliberately imports nothing but the stdlib -- it has to run in any interpreter,
including ones with no torch at all.

The headline is a per-group geomean, but the geomean is not the point. Every
regression in this project's motivating history (see README) was a case where
the aggregate improved and one cell got dramatically worse: mul's TensorIterator
migration, im2col's 64-bit indexing, F.linear's seq=1 decode path. So the
aggregate row always carries p10 and worst-case alongside it, and the
regressions table is a first-class output rather than an appendix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

#: A cell must be this much slower than the previous release before we call it a
#: regression, *and* it must clear the measured noise for that cell.
REGRESSION_RATIO = 1.15
NOISE_FLOOR = 0.03


def version_key(v: str) -> tuple:
    """Sort '2.9.0' < '2.14.0' < '2.15.0a0+git9a7fba2'."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?", v)
    if not m:
        return (999, 999, 999, 9, 0)
    major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
    # A pre-release of X sorts after every release of X-1 but before X itself.
    pre = m.group(4)
    return (major, minor, patch, 0 if pre else 1, int(m.group(5) or 0))


def load(results: Path, version_filter: set[str] | None = None) -> tuple[dict, list]:
    runs: dict[str, dict] = {}
    rows: list[dict] = []
    files = sorted(results.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl under {results}")
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "run":
                    runs[rec["run_id"]] = rec
                elif rec.get("type") == "result" and "error" not in rec:
                    rows.append(rec)
    for r in rows:
        run = runs[r["run_id"]]
        r["_version"] = run["torch"]["torch_version"]
        r["_env_id"] = run["machine"]["env_id"]
        r["_slug"] = run["machine"]["slug"]
        r["_pass"] = run.get("pass_index", 0)
    if version_filter:
        rows = [r for r in rows if r["_version"] in version_filter]
    return runs, rows


def collapse_passes(rows: list[dict]) -> dict:
    """(version, key, mode) -> {t, spread, n_passes}.

    Median across interleaved passes of each pass's own median. Two levels of
    median is the cheapest defence against a single thermally-throttled pass.
    """
    bucket: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for r in rows:
        k = (r["_version"], r["key"], r["mode"])
        bucket[k].append(r["timing"]["median_s"])
        meta.setdefault(k, r)
    out = {}
    for k, times in bucket.items():
        med = statistics.median(times)
        spread = (max(times) - min(times)) / med if med > 0 and len(times) > 1 else 0.0
        out[k] = {"t": med, "spread": spread, "n_passes": len(times), "row": meta[k]}
    return out


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated percentile.

    Naive index arithmetic collapses p10 onto the minimum for small samples,
    which silently turns the "tail" column into a duplicate of "worst" exactly
    when a group has few cells.
    """
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def summarize(collapsed: dict, versions: list[str], baseline: str) -> dict:
    """Per (group, mode, version): distribution of speedup vs baseline.

    Speedup > 1 means faster than baseline. Only cells the baseline also
    measured are included, so a newly-added op cannot inflate the trend.
    """
    keys_by_mode: dict[str, set] = defaultdict(set)
    for (ver, key, mode) in collapsed:
        if ver == baseline:
            keys_by_mode[mode].add(key)

    summary: dict = defaultdict(dict)
    for ver in versions:
        for mode, keys in keys_by_mode.items():
            per_group: dict[str, list[tuple[float, str]]] = defaultdict(list)
            for key in keys:
                base = collapsed.get((baseline, key, mode))
                cur = collapsed.get((ver, key, mode))
                if not base or not cur or cur["t"] <= 0:
                    continue
                per_group[cur["row"]["group"]].append((base["t"] / cur["t"], key))
            for group, pairs in per_group.items():
                sp = sorted(s for s, _ in pairs)
                summary[(group, mode)][ver] = {
                    "geomean": _geomean(sp),
                    "median": statistics.median(sp),
                    "p10": _percentile(sp, 0.10),
                    "worst": sp[0],
                    "worst_key": min(pairs)[1],
                    "n": len(sp),
                }
    return summary


def find_regressions(collapsed: dict, versions: list[str]) -> list[dict]:
    """Cells slower in version N than in version N-1, beyond measured noise."""
    out = []
    for prev, cur in zip(versions, versions[1:]):
        for (ver, key, mode), rec in collapsed.items():
            if ver != cur:
                continue
            before = collapsed.get((prev, key, mode))
            if not before or rec["t"] <= 0:
                continue
            ratio = rec["t"] / before["t"]          # > 1 means slower now
            noise = max(before["spread"], rec["spread"], NOISE_FLOOR)
            if ratio > max(REGRESSION_RATIO, 1 + 2 * noise):
                row = rec["row"]
                out.append({
                    "from": prev, "to": cur, "key": key, "mode": mode,
                    "slowdown": ratio, "noise": noise,
                    "group": row["group"], "op": row["op"],
                    "variant": row["variant"], "dtype": row["dtype"],
                    "bound": row["bound"],
                    "t_before_ms": before["t"] * 1e3, "t_after_ms": rec["t"] * 1e3,
                    "tags": row.get("tags") or {},
                })
    out.sort(key=lambda d: -d["slowdown"])
    return out


def find_dispatch_bound(collapsed: dict, versions: list[str], factor: float = 3.0) -> list[dict]:
    """Cells whose latency hugely exceeds their pipelined cost.

    A large latency/throughput ratio means the op's wall-clock is dominated by
    host-side work or a sync, not by the kernel. That is invisible to a
    throughput-only benchmark and is exactly how the seq=1 decode path in
    #189847 hides: the kernel is fine, the dispatch is not.
    """
    out = []
    latest = versions[-1]
    for (ver, key, mode), rec in collapsed.items():
        if ver != latest or mode != "latency":
            continue
        thr = collapsed.get((ver, key, "throughput"))
        if not thr or thr["t"] <= 0:
            continue
        ratio = rec["t"] / thr["t"]
        if ratio >= factor:
            row = rec["row"]
            out.append({
                "version": ver, "key": key, "ratio": ratio,
                "latency_ms": rec["t"] * 1e3, "throughput_ms": thr["t"] * 1e3,
                "group": row["group"], "op": row["op"], "variant": row["variant"],
                "dtype": row["dtype"], "bound": row["bound"],
            })
    out.sort(key=lambda d: -d["ratio"])
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt_speedup(x: float) -> str:
    return f"{x:.2f}x"


def _peak_drift(peaks_by_ver: dict) -> dict[str, float]:
    """Relative spread of each peak metric across versions.

    The probe measures the machine, so any spread here is drift -- almost always
    thermal -- and it bounds how small a difference the sweep can honestly
    resolve.
    """
    out = {}
    for metric in ("bandwidth_gbps", "gflops_fp32", "gflops_fp16"):
        vals = [pk[metric] for pk in peaks_by_ver.values()
                if pk.get(metric) and pk.get("method") != "skipped"]
        if len(vals) > 1 and min(vals) > 0:
            out[metric] = (max(vals) - min(vals)) / max(vals)
    return out


def render(summary, regressions, dispatch, versions, baseline, runs, md: bool) -> str:
    L: list[str] = []
    h1, h2, sep = ("# ", "## ", "") if md else ("", "", "=" * 78)

    sample = next(iter(runs.values()))
    mach = sample["machine"]
    L += [f"{h1}pytorch_metal_perf", ""]
    L += [f"**Machine:** {mach['chip']} / {mach['gpu_name']} ({mach['gpu_cores']} GPU cores)"
          if md else f"Machine : {mach['chip']} / {mach['gpu_name']} ({mach['gpu_cores']} cores)"]
    L += [f"**OS:** {mach['os_product']} {mach['os_version']} build {mach['os_build']}"
          if md else f"OS      : {mach['os_product']} {mach['os_version']} ({mach['os_build']})"]
    mpp = mach["frameworks"].get("MetalPerformancePrimitives")
    L += [f"**MetalPerformancePrimitives:** {mpp or 'not present'}"
          if md else f"MPP     : {mpp or 'not present'}"]
    L += [f"**Baseline:** torch {baseline}" if md else f"Baseline: torch {baseline}", ""]

    # Per-version peak probes: these should agree. If they do not, the machine
    # changed state mid-sweep and the whole series is suspect.
    L += [f"{h2}Measured device peaks (sanity check -- should not vary with torch)", ""]
    peaks_by_ver = {}
    for run in runs.values():
        peaks_by_ver.setdefault(run["torch"]["torch_version"], run["peaks"])
    if md:
        L += ["| torch | bandwidth GB/s | fp32 GFLOP/s | fp16 GFLOP/s |",
              "|---|---:|---:|---:|"]
    for v in versions:
        pk = peaks_by_ver.get(v)
        if not pk:
            continue
        if md:
            L.append(f"| {v} | {pk['bandwidth_gbps']:.0f} | {pk['gflops_fp32']:.0f} "
                     f"| {pk['gflops_fp16']:.0f} |")
        else:
            L.append(f"  {v:<22} {pk['bandwidth_gbps']:7.0f} GB/s  "
                     f"{pk['gflops_fp32']:7.0f} / {pk['gflops_fp16']:7.0f} GFLOP/s")
    L += [""]

    # The probe measures the machine, not torch. Spread across versions is a
    # direct estimate of how much thermal drift contaminated the whole sweep.
    drift = _peak_drift(peaks_by_ver)
    if drift:
        worst = max(drift.values())
        msg = ("Peak probe varied by "
               + ", ".join(f"{v:.0%} ({k})" for k, v in sorted(drift.items()))
               + " across versions.")
        if worst > 0.05:
            msg += (" That exceeds 5%: the GPU was throttling, so differences below "
                    "roughly this magnitude are drift, not torch. Raise --cooldown "
                    "or --passes and re-run.")
        L += [(f"> **Note.** {msg}" if md else f"  NOTE: {msg}"), ""]

    L += [f"{h2}Speedup vs torch {baseline} (higher is better)", ""]
    for mode in ("latency", "throughput"):
        groups = sorted({g for (g, m) in summary if m == mode})
        if not groups:
            continue
        L += [f"### {mode}" if md else f"-- {mode} --", ""]
        if md:
            L += ["| group | stat | " + " | ".join(versions) + " |",
                  "|---|---|" + "---:|" * len(versions)]
        else:
            L.append(f"  {'group':<12} {'stat':<8} " +
                     " ".join(f"{v[:12]:>12}" for v in versions))
        for g in groups:
            per_ver = summary[(g, mode)]
            n = per_ver.get(versions[-1], {}).get("n", 0)
            for stat in ("geomean", "p10", "worst"):
                cells = [_fmt_speedup(per_ver[v][stat]) if v in per_ver else "-"
                         for v in versions]
                label = f"{g} (n={n})" if stat == "geomean" else ""
                if md:
                    L.append(f"| {label} | {stat} | " + " | ".join(cells) + " |")
                else:
                    L.append(f"  {label:<12} {stat:<8} " +
                             " ".join(f"{c:>12}" for c in cells))
            if not md:
                L.append("")
        L.append("")

    L += [f"{h2}Regressions vs previous version", ""]
    if not regressions:
        L += ["_none above the noise threshold_" if md else "  (none above threshold)", ""]
    else:
        if md:
            L += ["| slowdown | from | to | group | op | variant | dtype | bound | mode | before ms | after ms |",
                  "|---:|---|---|---|---|---|---|---|---|---:|---:|"]
        for r in regressions[:40]:
            if md:
                L.append(
                    f"| **{r['slowdown']:.2f}x** | {r['from']} | {r['to']} | {r['group']} "
                    f"| {r['op']} | {r['variant']} | {r['dtype']} | {r['bound']} | {r['mode']} "
                    f"| {r['t_before_ms']:.3f} | {r['t_after_ms']:.3f} |")
            else:
                L.append(f"  {r['slowdown']:5.2f}x slower  {r['from']} -> {r['to']}  "
                         f"{r['op']}/{r['variant']}/{r['dtype']}/{r['bound']}/{r['mode']}  "
                         f"({r['t_before_ms']:.3f} -> {r['t_after_ms']:.3f} ms)")
        L.append("")

    L += [f"{h2}Dispatch-bound cells (latency >= 3x pipelined cost, torch {versions[-1]})", ""]
    L += ["Host-side overhead or a hidden sync, not kernel time. Invisible to a "
          "throughput-only benchmark.", ""]
    if not dispatch:
        L += ["_none_" if md else "  (none)", ""]
    else:
        if md:
            L += ["| ratio | group | op | variant | dtype | bound | latency ms | pipelined ms |",
                  "|---:|---|---|---|---|---|---:|---:|"]
        for r in dispatch[:25]:
            if md:
                L.append(f"| {r['ratio']:.1f}x | {r['group']} | {r['op']} | {r['variant']} "
                         f"| {r['dtype']} | {r['bound']} | {r['latency_ms']:.3f} "
                         f"| {r['throughput_ms']:.3f} |")
            else:
                L.append(f"  {r['ratio']:5.1f}x  {r['op']}/{r['variant']}/{r['dtype']}/"
                         f"{r['bound']}  ({r['latency_ms']:.3f} vs {r['throughput_ms']:.3f} ms)")
        L.append("")

    if sep:
        L.insert(0, sep)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analyze")
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--baseline", default=None, help="default: lowest version present")
    p.add_argument("--versions", nargs="*", default=None)
    p.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--allow-mixed-env", action="store_true",
                   help="compare across different machines/OS builds (almost never right)")
    args = p.parse_args(argv)

    runs, rows = load(args.results, set(args.versions) if args.versions else None)
    if not rows:
        raise SystemExit("no successful results found")

    envs = {r["_env_id"] for r in rows}
    if len(envs) > 1 and not args.allow_mixed_env:
        slugs = sorted({r["_slug"] for r in rows})
        raise SystemExit(
            "results span multiple machine/OS environments:\n  "
            + "\n  ".join(slugs)
            + "\n\nmacOS gates which kernels MPS selects, so these are not comparable.\n"
              "Re-run the baseline on the current OS, or pass --allow-mixed-env if you\n"
              "genuinely want to measure the OS difference."
        )

    collapsed = collapse_passes(rows)
    versions = sorted({r["_version"] for r in rows}, key=version_key)
    baseline = args.baseline or versions[0]
    if baseline not in versions:
        raise SystemExit(f"baseline {baseline} not in results: {versions}")

    summary = summarize(collapsed, versions, baseline)
    regressions = find_regressions(collapsed, versions)
    dispatch = find_dispatch_bound(collapsed, versions)

    if args.format == "json":
        payload = {
            "machine": next(iter(runs.values()))["machine"],
            "peaks_by_version": {
                r["torch"]["torch_version"]: r["peaks"] for r in runs.values()
            },
            "versions": versions,
            "baseline": baseline,
            "summary": {f"{g}|{m}": v for (g, m), v in summary.items()},
            "regressions": regressions,
            "dispatch_bound": dispatch,
        }
        text = json.dumps(payload, indent=2)
    else:
        text = render(summary, regressions, dispatch, versions, baseline, runs,
                      md=args.format == "markdown")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
