# Regression corpus

One file per historically-observed MPS performance regression, each pinned to
the PR that caused it and (where one exists) the PR that fixed it, with the
release each landed in **verified against the release tags** rather than
inferred from commit dates:

```
for t in v2.9.0 v2.10.0 v2.11.0 v2.12.1 v2.13.0 v2.14.0; do
  git log --oneline "$t" --grep="#<PR>)" -1 && break
done
```

## Why this exists

Every entry here is a case where an MPS change made the aggregate better and one
specific cell much worse. That is the failure mode this whole repo is built to
catch, so the corpus doubles as the suite's own test set: running the sweep
across the version matrix should show each entry's `detect` cell degrading at
`caused_in` and recovering at `fixed_in`. If it does not, the sweep is not
sensitive enough and adding more ops will not help.

## Pattern codes

| code | pattern |
|---|---|
| A | launch overhead traded for kernel throughput (MPSGraph caching vs per-op Metal encoders) |
| B | precision widening -- half inputs computed in fp32 |
| C | index-width inflation -- 64-bit math added for >2^32 correctness |
| D | lost specialization on non-dense inputs |
| E | shape-regime cliff at a heuristic threshold |
| F | memory-format assumption (channels_last) |
| G | silent extra copy, usually `.contiguous()` on a grad path |
| H | hidden CPU<->GPU sync inside the kernel |
| I | misaligned storage offset |
| J | scalar / broadcast fast-path divergence |

## Schema

- `detect` names the sweep cell that should expose it, in the same
  `group/op/variant/dtype/shape/bound` form as `Case.key`.
- `status: open` means the cell is still bad on trunk as of the noted date.
