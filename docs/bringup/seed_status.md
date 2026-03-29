# Seed Kernel Gate Status

This page summarizes the current interpreted gate state of the initial seed kernels
under the PTO 910B superproject bring-up plan.

Legend:
- `G2`: PTO compile
- `G3`: baseline/reference contract
- `G4`: correctness
- `G5`: benchmark
- `G6`: regression
- `G8`: performance convergence

As of the current local bring-up state, all six seed kernels have working bounded
slices that are compile-green, correctness-green, benchmarkable, and visible in
`bench/generated/*/report.json`. The main remaining work is performance closure
and family-expansion-driven cross-stack hardening.

## Current seed summary

| Kernel | G2 | G3 | G4 | G5 | G6 | G8 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `posembedding/apply_rotary_pos_emb` | pass | pass | pass | pass | pass | open | PTO slice is slower than baseline; still blocked on broader posembedding/vector pipeline generalization. |
| `gmm/grouped_matmul` | pass | pass | pass | pass | pass | open | PTO slice is correctness-green but still slower than baseline; cube preload/pipeline surface remains the main cross-stack gap. |
| `ffn/ffn` | pass | pass | pass | pass | pass | open | Runnable dense staged FFN seed exists; performance remains far from baseline and should drive next family-level tuning work. |
| `moe/moe_token_permute` | pass | pass | pass | pass | pass | open | Bounded top-1 permutation slice is closed functionally; performance and broader semantics remain open. |
| `attention/flash_attention_score` | pass | pass | pass | pass | pass | open | Dense staged attention seed is closed functionally; later work is generalized attention semantics and performance. |
| `mc2/matmul_reduce_scatter` | pass | pass | pass | pass | pass | open | Local PTO math + host collective harness is working; next work is distributed-family expansion and performance closure. |

## Interpretation

These seed kernels should currently be treated as **phase-1 functional closure achieved,
performance closure pending**.

That means:

- they should no longer be treated as not-yet-runnable seeds,
- checklist state should indicate partial closure (`[~]`) rather than untouched (`[ ]`),
- the next wave of work should prioritize performance and reusable family/cross-stack surfaces
  rather than rediscovering basic bring-up for these same kernels.
