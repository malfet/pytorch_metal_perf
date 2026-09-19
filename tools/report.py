#!/usr/bin/env python3
"""Render `analyze.py --format json` into a self-contained HTML report.

Stdlib only, no external assets, so the output file can be dropped into a PR
comment or opened straight from disk.

    python tools/analyze.py --format json --out reports/data.json
    python tools/report.py reports/data.json --out reports/index.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e3e3e3;
  --head: #f6f6f7; --win: #0b7a3b; --loss: #b3261e; --flat: #666;
  --win-bg: #e8f5ed; --loss-bg: #fdecea;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181a; --fg: #e8e8e8; --muted: #9aa0a6; --line: #2e3236;
    --head: #1e2124; --win: #4ade80; --loss: #f87171; --flat: #9aa0a6;
    --win-bg: #14301f; --loss-bg: #3a1a19;
  }
}
:root[data-theme="dark"] {
  --bg: #16181a; --fg: #e8e8e8; --muted: #9aa0a6; --line: #2e3236;
  --head: #1e2124; --win: #4ade80; --loss: #f87171; --flat: #9aa0a6;
  --win-bg: #14301f; --loss-bg: #3a1a19;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0; padding: 2rem 1.25rem 5rem;
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .4rem; letter-spacing: -.01em; }
p.note { color: var(--muted); margin: .2rem 0 1rem; font-size: .9rem; max-width: 62ch; }
.meta { color: var(--muted); font-size: .9rem; margin-bottom: 1.5rem; }
.meta b { color: var(--fg); font-weight: 600; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; min-width: 640px; }
th, td { padding: .38rem .6rem; text-align: right; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child,
th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { background: var(--head); font-weight: 600; position: sticky; top: 0; }
tbody tr:hover td { background: var(--head); }
.num { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
.win { color: var(--win); font-weight: 600; }
.loss { color: var(--loss); font-weight: 600; }
.flat { color: var(--flat); }
td.win-c { background: var(--win-bg); }
td.loss-c { background: var(--loss-bg); }
.stat { color: var(--muted); font-size: .82rem; }
.empty { color: var(--muted); font-style: italic; }
code { font: 12.5px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def cls(speedup: float) -> str:
    if speedup >= 1.10:
        return "win"
    if speedup <= 0.90:
        return "loss"
    return "flat"


def _table(headers, rows, aligns=None) -> str:
    out = ['<div class="scroll"><table><thead><tr>']
    out += [f"<th>{html.escape(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def build(data: dict) -> str:
    m = data["machine"]
    versions = data["versions"]
    baseline = data["baseline"]
    P = []

    P.append(f"<h1>MPS operator performance, torch {versions[0]} → {versions[-1]}</h1>")
    mpp = m["frameworks"].get("MetalPerformancePrimitives") or "not present"
    P.append(
        '<div class="meta">'
        f"<b>{html.escape(m['chip'])}</b> &middot; {html.escape(m['gpu_name'] or '')} "
        f"({m['gpu_cores']} GPU cores) &middot; "
        f"<b>{html.escape(m['os_product'])} {html.escape(m['os_version'])}</b> "
        f"build {html.escape(m['os_build'])} &middot; "
        f"MetalPerformancePrimitives {html.escape(str(mpp))}<br>"
        f"Baseline torch {html.escape(baseline)}. "
        "Every number below is from this one OS build — macOS gates which kernels "
        "MPS selects, so results from a different build are a different series."
        "</div>"
    )

    # --- peaks ---
    P.append("<h2>Measured device peaks</h2>")
    P.append('<p class="note">Probed independently in each environment. These should '
             "agree across torch versions; if they drift, the machine changed state "
             "mid-sweep and every comparison below is suspect.</p>")
    rows = []
    for v in versions:
        pk = data["peaks_by_version"].get(v)
        if not pk:
            continue
        rows.append([
            f"<td>{html.escape(v)}</td>",
            f'<td class="num">{pk["bandwidth_gbps"]:.0f}</td>',
            f'<td class="num">{pk["gflops_fp32"]:.0f}</td>',
            f'<td class="num">{pk["gflops_fp16"]:.0f}</td>',
        ])
    P.append(_table(["torch", "GB/s", "fp32 GFLOP/s", "fp16 GFLOP/s"], rows))

    # --- per-group trend ---
    P.append(f"<h2>Speedup vs torch {html.escape(baseline)}</h2>")
    P.append('<p class="note">Higher is better. <b>geomean</b> is the headline trend; '
             "<b>p10</b> and <b>worst</b> are the ones that matter. Every regression this "
             "suite was built for looked like an improving geomean next to a collapsing "
             "worst case.</p>")
    for mode in ("latency", "throughput"):
        keys = sorted(k for k in data["summary"] if k.endswith(f"|{mode}"))
        if not keys:
            continue
        P.append(f"<h3 style='font-size:1rem;margin:1.2rem 0 .3rem'>{mode}</h3>")
        rows = []
        for k in keys:
            group = k.split("|")[0]
            per_ver = data["summary"][k]
            n = per_ver.get(versions[-1], {}).get("n", 0)
            for i, stat in enumerate(("geomean", "p10", "worst")):
                cells = [
                    f"<td>{html.escape(group)} <span class='stat'>n={n}</span></td>"
                    if i == 0 else "<td></td>",
                    f'<td class="stat">{stat}</td>',
                ]
                for v in versions:
                    if v not in per_ver:
                        cells.append('<td class="num flat">&ndash;</td>')
                        continue
                    s = per_ver[v][stat]
                    c = cls(s)
                    bg = " win-c" if c == "win" else (" loss-c" if c == "loss" else "")
                    cells.append(f'<td class="num {c}{bg}">{s:.2f}x</td>')
                rows.append(cells)
        P.append(_table(["group", "stat", *versions], rows))

    # --- regressions ---
    regs = data["regressions"]
    P.append("<h2>Regressions vs the previous version</h2>")
    P.append('<p class="note">Cells slower than in the preceding release by more than '
             "15% <em>and</em> more than twice the measured across-pass spread for that "
             "cell.</p>")
    if not regs:
        P.append('<p class="empty">None above the noise threshold.</p>')
    else:
        rows = []
        for r in regs[:50]:
            rows.append([
                f'<td class="num loss">{r["slowdown"]:.2f}x</td>',
                f"<td><code>{html.escape(r['op'])}/{html.escape(r['variant'])}</code></td>",
                f"<td>{html.escape(r['group'])}</td>",
                f"<td>{html.escape(r['dtype'])}</td>",
                f"<td>{html.escape(r['bound'])}</td>",
                f"<td>{html.escape(r['mode'])}</td>",
                f"<td>{html.escape(r['from'])} &rarr; {html.escape(r['to'])}</td>",
                f'<td class="num">{r["t_before_ms"]:.3f}</td>',
                f'<td class="num">{r["t_after_ms"]:.3f}</td>',
            ])
        P.append(_table(
            ["slowdown", "op/variant", "group", "dtype", "bound", "mode",
             "versions", "before ms", "after ms"], rows))

    # --- dispatch-bound ---
    disp = data["dispatch_bound"]
    P.append("<h2>Dispatch-bound cells</h2>")
    P.append('<p class="note">Latency at least 3x the pipelined cost on torch '
             f"{html.escape(versions[-1])}: wall-clock dominated by host-side work or a "
             "sync rather than by the kernel. A throughput-only benchmark reports these "
             "as healthy.</p>")
    if not disp:
        P.append('<p class="empty">None.</p>')
    else:
        rows = []
        for r in disp[:30]:
            rows.append([
                f'<td class="num loss">{r["ratio"]:.1f}x</td>',
                f"<td><code>{html.escape(r['op'])}/{html.escape(r['variant'])}</code></td>",
                f"<td>{html.escape(r['group'])}</td>",
                f"<td>{html.escape(r['dtype'])}</td>",
                f"<td>{html.escape(r['bound'])}</td>",
                f'<td class="num">{r["latency_ms"]:.3f}</td>',
                f'<td class="num">{r["throughput_ms"]:.3f}</td>',
            ])
        P.append(_table(
            ["latency/pipelined", "op/variant", "group", "dtype", "bound",
             "latency ms", "pipelined ms"], rows))

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>pytorch_metal_perf</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        + "".join(P)
        + "</div></body></html>"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="report")
    p.add_argument("json", type=Path, help="output of `analyze.py --format json`")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    data = json.loads(args.json.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(data))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
