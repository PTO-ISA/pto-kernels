# Phase 1 Seed Kernels

## Common Checklist

- [ ] Baseline adapter exists
- [ ] ops-transformer seed package built or current compile/install blocker recorded
- [ ] PTO-DSL source exists
- [x] Manual event/wait pairs removed from active PTO seed kernels; `ptoas` owns sync insertion
- [ ] `kernel.pto` archived
- [ ] `kernel.cpp` archived
- [ ] `.so` archived
- [ ] Correctness report stored
- [ ] Benchmark report stored
- [ ] Gap board updated
- [ ] Canonical flags confirmed or exception documented

## Seed Kernels

- [x] `posembedding/apply_rotary_pos_emb`
  Baseline and PTO seeds both run for constrained `TND + BSND + half + fp16 + D=128` variants; the active remaining gap is broader rotary-mode and family-level posembedding generalization.
- [x] `gmm/grouped_matmul`
  Baseline and PTO seeds both run on the BF16 output contract; the active remaining gap is routing/group-list semantics and PTO-side performance tuning.
- [x] `ffn/ffn`
  Baseline and PTO seeds now run for dense `fp16 + relu + no-bias`; the staged PTO implementation is now factored through a shared FFN helper, and the active remaining gap is fused cube-vector-cube lowering plus PTO-side performance closure.
- [x] `moe/moe_token_permute`
  Baseline and PTO seeds now run for top-1 permutation; the PTO seed currently uses a host-precomputed gather map, and the remaining gap is on-device sort/inverse-permutation routing support.
- [x] `attention/flash_attention_score`
  Baseline and PTO seeds now run for dense `BNSD` fp16 attention; the staged PTO path is now factored through a shared dense-attention helper, and the remaining gap is masked/online attention generalization plus performance closure.
- [x] `mc2/matmul_reduce_scatter`
  Baseline and PTO seeds now run on a local 2-rank HCCL harness; the PTO seed currently covers only the local matmul and relies on a host-orchestrated HCCL all-reduce plus row chunking contract instead of PTODSL MC2 collectives.

## Sync Ownership

- [x] `grouped_matmul`, `apply_rotary_pos_emb`, `ffn`, `flash_attention_score`, and `matmul_reduce_scatter` no longer emit DSL-level event record/wait pairs in the PTO source.
- [x] Current generated artifacts confirm `ptoas --enable-insert-sync` inserts the required sync edges for the runnable seed kernels.
- [ ] Remaining family ports should stay on the same contract and only add explicit barriers when they are semantically required outside the `ptoas` insert-sync scope.
