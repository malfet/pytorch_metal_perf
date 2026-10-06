#!/usr/bin/env python3
"""Render reports/workloads/*.json (from tools/trace_models.py) into WORKLOADS.md.

Stdlib only, like tools/analyze.py.

"Traffic" throughout is the bytes an op's tensors span: every input plus every
output, except gather-style ops (embedding, index_select, ...) which are charged
for the rows they read rather than the whole table. It is a proxy for where a
memory-bound backend spends its time, not a measurement: a matmul's FLOPs and a
pointwise op's launch cost are both invisible to it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAMILIES = ("llm", "encoder", "vision", "audio", "diffusion", "video", "preprocessor")
FAMILY_TITLE = {"llm": "LLMs", "encoder": "Encoder / small transformers", "vision": "Vision",
                "audio": "Audio", "diffusion": "Diffusion (image)", "video": "Video diffusion",
                "preprocessor": "ControlNet preprocessors"}

CATEGORY_OPS = {
    "matmul": {"linear", "linear_backward", "mm", "addmm", "bmm", "baddbmm", "matmul", "mv", "addmv", "_addmm_activation"},
    "conv": {"convolution", "_convolution", "convolution_backward", "conv2d", "conv1d"},
    "attention": {"_scaled_dot_product_attention_math_for_mps", "_scaled_dot_product_attention_math",
                  "_scaled_dot_product_flash_attention_for_cpu", "scaled_dot_product_attention",
                  "_scaled_dot_product_fused_attention_overrideable",
                  "_scaled_dot_product_attention_math_for_mps_backward"},
    "norm": {"native_layer_norm", "native_layer_norm_backward", "native_batch_norm",
             "_native_batch_norm_legit", "_native_batch_norm_legit_no_training",
             "native_batch_norm_backward", "native_group_norm", "native_group_norm_backward",
             "_fused_rms_norm", "_fused_rms_norm_backward", "batch_norm_backward"},
    "softmax": {"_softmax", "_safe_softmax", "_log_softmax", "_softmax_backward_data", "_log_softmax_backward_data"},
    "copy/layout": {"_to_copy", "copy_", "clone", "cat", "stack", "constant_pad_nd", "contiguous",
                    "roll", "flip", "repeat", "_unsafe_index", "upsample_nearest2d",
                    "upsample_bilinear2d", "pixel_shuffle", "tril", "triu", "fill_", "zero_",
                    "zeros", "ones", "full", "zeros_like", "ones_like", "full_like", "arange",
                    "scalar_tensor", "_local_scalar_dense", "new_zeros", "new_full", "new_ones",
                    "slice_backward", "select_backward"},
    "gather/scatter": {"embedding", "embedding_dense_backward", "index", "index_select", "gather",
                       "scatter", "scatter_add", "index_put", "index_put_", "_index_put_impl_",
                       "masked_select", "take", "index_add_"},
    "pooling": {"max_pool2d", "max_pool2d_with_indices", "avg_pool2d", "adaptive_avg_pool2d", "_adaptive_avg_pool2d",
                "max_pool2d_with_indices_backward", "_adaptive_avg_pool2d_backward",
                "avg_pool2d_backward"},
    "loss": {"nll_loss_forward", "nll_loss_backward", "nll_loss2d_forward", "mse_loss",
             "binary_cross_entropy", "binary_cross_entropy_with_logits"},
    "detection": {"torchvision::roi_align", "torchvision::nms", "nonzero"},
    # Pointwise ops aten does not tag as pointwise.
    "elementwise": {"hardswish", "hardsigmoid", "hardswish_backward", "gelu_backward",
                    "silu_backward", "sigmoid_backward", "tanh_backward"},
}
# Optimizer and RNG ops are recognised by prefix.
PREFIX_CATEGORIES = (("_foreach_", "optimizer"), ("_fused_adam", "optimizer"),
                     ("uniform", "rng"), ("normal", "rng"), ("bernoulli", "rng"),
                     ("randint", "rng"), ("rand", "rng"))

# What pmp/ops benchmarks today, by aten base name. "partial" means the op is in
# the suite but the shapes/layouts models use are not (see the gap list).
PMP_COVERAGE = {
    **{op: "binary" for op in ("add", "mul", "div", "maximum", "pow", "atan2", "lt")},
    **{op: "unary" for op in ("neg", "abs", "sqrt", "rsqrt", "exp", "sin", "erf", "sigmoid",
                              "tanh", "isnan")},
    **{op: "reduction" for op in ("sum", "mean", "max", "argmax", "amin", "linalg_vector_norm")},
    **{op: "normalization" for op in ("native_layer_norm", "_softmax", "_log_softmax",
                                      "_fused_rms_norm")},
    **{op: "matmul" for op in ("linear", "mm", "mv", "addmm")},
}


def base(op: str) -> str:
    """'mul.Tensor' -> 'mul', 'add_.Tensor' -> 'add' (in-place shares the kernel)."""
    name = op.split(".")[0]
    if name.endswith("_") and not name.startswith("_") and name not in ("copy_", "fill_", "zero_"):
        name = name[:-1]
    return name


def category(op: str, tag: str | None) -> str:
    b = op.split(".")[0]
    for cat, ops in CATEGORY_OPS.items():
        if b in ops or base(op) in ops:
            return cat
    for prefix, cat in PREFIX_CATEGORIES:
        if b.startswith(prefix):
            return cat
    if tag == "pointwise":
        return "elementwise"
    if tag == "reduction" or base(op) in ("sum", "mean", "amax", "amin", "max", "min", "argmax",
                                          "argmin", "var_mean", "var", "std", "any", "all",
                                          "linalg_vector_norm", "norm", "cumsum", "prod",
                                          "logsumexp", "topk", "sort"):
        return "reduction"
    return "other"


def mib(n: float) -> str:
    v = n / 2**20
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}" if v >= 0.1 else "<0.1"


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def layouts_of(sig: str) -> list[str]:
    return re.findall(r"\{(\w+)\}", sig)


def dtype_of(sig: str) -> str:
    return sig.split("[")[0]


def short_shapes(inputs: list[str], limit: int = 4) -> str:
    s = " · ".join(inputs[:limit])
    if len(inputs) > limit:
        s += f" · …(+{len(inputs) - limit})"
    return s


def load(dirpath: Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(dirpath.glob("*.json"))]
    order = {f: i for i, f in enumerate(FAMILIES)}
    return sorted(runs, key=lambda r: (order.get(r["family"], 99), r["model"]))


def render(runs: list[dict], top: int) -> str:
    L: list[str] = []
    torch_versions = sorted({r["torch"] for r in runs})
    L += [
        "# Workloads: what real networks run on MPS",
        "",
        "Generated by `tools/trace_models.py` (trace) and `tools/workloads_md.py` (this",
        "page); do not edit by hand. Raw per-model data is in `reports/workloads/*.json`.",
        "",
        "Each model runs eagerly on MPS, in the dtype it ships in, under a `TorchDispatchMode`",
        "that logs every non-view aten op with its input shapes, dtypes, layouts and small",
        "non-tensor args (reduction dims, conv strides). Weights are random — shapes do not",
        "depend on them. The point is to check the benchmark suite's hand-picked shapes",
        "against what networks actually execute.",
        "",
        "**Reading the tables.** *Traffic* is the bytes an op's input and output tensors span",
        "(gather-style ops are charged for the rows read, not the whole table). It is a proxy",
        "for time on a memory-bound backend, not a measurement: it understates matmul and",
        "conv, whose cost is FLOPs, and every launch-bound small op. Shapes are written",
        "`dtype[dims]`; a suffix marks a non-contiguous input:",
        "",
        "| suffix | meaning |",
        "|---|---|",
        "| `{T}` | last two dims swapped (`x.t()`, `k.transpose(-1, -2)`) |",
        "| `{perm}` | other permutation of a dense tensor (e.g. attention head shuffles) |",
        "| `{cl}` | channels_last 4D |",
        "| `{bcast}` | broadcast (a stride is 0) |",
        "| `{strided}` | has gaps — a slice such as `x[..., ::2]` or a narrowed dim |",
        "",
        f"Traced with torch {', '.join(torch_versions)}. Depth: models marked *2 of N layers*",
        "ran with two decoder layers; every layer has identical shapes, so only call counts",
        "and traffic scale.",
        "",
    ]

    # -- model table ------------------------------------------------------------
    L += ["## Models", "",
          "| family | model | dtype | input | phase | op calls | signatures | traffic MiB |",
          "|---|---|---|---|---|---:|---:|---:|"]
    for r in runs:
        depth = ""
        if r.get("layers_traced") and r["layers_traced"] != r.get("layers_full"):
            depth = f" (2 of {r['layers_full']} layers)"
        for ph, d in r["phases"].items():
            tot = sum(o["bytes"] for o in d["ops"])
            L.append(f"| {r['family']} | {r['model']}{depth} | {r['dtype']} | "
                     f"{r.get('input', '')} | {ph} | {d['calls']:,} | {len(d['ops'])} | {mib(tot)} |")
    L.append("")

    # -- category mix ------------------------------------------------------------
    cats = ["matmul", "conv", "attention", "elementwise", "norm", "softmax", "reduction",
            "copy/layout", "gather/scatter", "pooling", "loss", "detection", "optimizer", "rng",
            "other"]
    L += ["## Where the traffic goes", "",
          "Share of traffic by op category, per model and phase. `elementwise` is everything",
          "tagged pointwise in aten (add, mul, silu, gelu, where, …).", ""]
    present = set()
    rows = []
    for r in runs:
        for ph, d in r["phases"].items():
            agg = defaultdict(float)
            for o in d["ops"]:
                agg[category(o["op"], o.get("tag"))] += o["bytes"]
            present |= {c for c, v in agg.items() if v}
            rows.append((r, ph, agg, sum(agg.values())))
    cols = [c for c in cats if c in present]
    L.append("| model | phase | " + " | ".join(cols) + " |")
    L.append("|---|---|" + "---:|" * len(cols))
    for r, ph, agg, tot in rows:
        L.append(f"| {r['model']} | {ph} | " +
                 " | ".join(pct(agg[c], tot) if agg[c] >= 0.005 * tot else "" for c in cols) + " |")
    L.append("")

    # -- dtype and layout --------------------------------------------------------
    L += ["## Dtypes and layouts", "",
          "*Upcast* is the share of traffic in ops whose first input is fp32 inside a model",
          "that ships in fp16/bf16 — norms and softmax computed in fp32, mostly. The layout",
          "columns are shares of traffic in ops with at least one input of that layout.", ""]
    lay_cols = ["T", "perm", "cl", "bcast", "strided"]
    L.append("| model | phase | dtype | upcast to fp32 | " + " | ".join(f"`{{{c}}}`" for c in lay_cols) + " |")
    L.append("|---|---|---|---:|" + "---:|" * len(lay_cols))
    for r in runs:
        for ph, d in r["phases"].items():
            tot = sum(o["bytes"] for o in d["ops"]) or 1
            up = ""
            if r["dtype"] in ("float16", "bfloat16"):
                up = pct(sum(o["bytes"] for o in d["ops"] if o["inputs"] and
                             dtype_of(o["inputs"][0]) == "f32"), tot)
            lay = []
            for c in lay_cols:
                b = sum(o["bytes"] for o in d["ops"] if any(c in layouts_of(i) for i in o["inputs"]))
                lay.append(pct(b, tot) if b >= 0.005 * tot else "")
            L.append(f"| {r['model']} | {ph} | {r['dtype']} | {up} | " + " | ".join(lay) + " |")
    L.append("")

    # -- coverage ----------------------------------------------------------------
    by_op: dict[str, dict] = {}
    grand = 0.0
    for r in runs:
        for ph, d in r["phases"].items():
            tot = sum(o["bytes"] for o in d["ops"]) or 1
            grand += 1
            for o in d["ops"]:
                b = base(o["op"])
                e = by_op.setdefault(b, {"share": 0.0, "models": set(), "dtypes": set(),
                                         "layouts": set(), "cat": category(o["op"], o.get("tag")),
                                         "example": None, "ex_bytes": 0})
                e["share"] += o["bytes"] / tot
                e["models"].add(r["model"])
                e["dtypes"] |= {dtype_of(i) for i in o["inputs"]}
                e["layouts"] |= {lay for i in o["inputs"] for lay in layouts_of(i)}
                if o["bytes"] > e["ex_bytes"]:
                    e["ex_bytes"] = o["bytes"]
                    e["example"] = (r["model"], o)
    L += ["## Coverage against the benchmark suite", "",
          "Every op base name seen, ranked by its mean share of traffic across all traced",
          "model-phases. *pmp* is the benchmark group that covers the op today; blank means",
          "the suite does not measure it. *Example* is the single heaviest signature.", "",
          "| op | category | mean share | models | dtypes | non-dense layouts | pmp | example |",
          "|---|---|---:|---:|---|---|---|---|"]
    ranked = sorted(by_op.items(), key=lambda kv: -kv[1]["share"])
    for name, e in ranked:
        share = e["share"] / grand
        if share < 0.001:
            continue
        m, o = e["example"]
        ex = f"{m}: `{short_shapes(o['inputs'], 3)}`"
        L.append(f"| `{name}` | {e['cat']} | {100 * share:.1f}% | {len(e['models'])} | "
                 f"{', '.join(sorted(e['dtypes']))} | {', '.join(sorted(e['layouts']))} | "
                 f"{PMP_COVERAGE.get(name, '')} | {ex} |")
    L.append("")

    # -- per model ---------------------------------------------------------------
    L += ["## Per-model detail", "",
          f"The {top} heaviest op signatures of each phase, by traffic.", ""]
    for fam in FAMILIES:
        fam_runs = [r for r in runs if r["family"] == fam]
        if not fam_runs:
            continue
        L += [f"### {FAMILY_TITLE[fam]}", ""]
        for r in fam_runs:
            depth = ""
            if r.get("layers_traced") and r["layers_traced"] != r.get("layers_full"):
                depth = f", 2 of {r['layers_full']} layers"
            note = f" — {r['note']}" if r.get("note") else ""
            L += [f"#### {r['model']} ({r['dtype']}, {r.get('input', '')}{depth}){note}", ""]
            for ph, d in r["phases"].items():
                tot = sum(o["bytes"] for o in d["ops"]) or 1
                L += [f"**{ph}** — {d['calls']:,} op calls, {mib(tot)} MiB traffic", "",
                      "| calls | share | op | inputs | args |", "|---:|---:|---|---|---|"]
                for o in d["ops"][:top]:
                    args = ", ".join(o["args"])
                    L.append(f"| {o['count']} | {pct(o['bytes'], tot)} | `{o['op']}` | "
                             f"`{short_shapes(o['inputs'])}` | {('`' + args + '`') if args else ''} |")
                L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="workloads_md")
    p.add_argument("--data", type=Path, default=REPO / "reports" / "workloads")
    p.add_argument("--out", type=Path, default=REPO / "WORKLOADS.md")
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args(argv)
    args.out.write_text(render(load(args.data), args.top))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
