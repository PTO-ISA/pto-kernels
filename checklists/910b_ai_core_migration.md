# 910B AI Core Migration Checklist

Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


Legend:
- `[x]` baseline and PTO both stable, or PTO is stable and the remaining blocker is external to PTO
- `[~]` PTO-owned blocker still open, or the kernel is still scalar-heavy
- `[ ]` not started


## Wave 1

- [x] `posembedding/apply_rotary_pos_emb`
- [x] `posembedding/dequant_rope_quant_kvcache`
- [x] `posembedding/interleave_rope`
- [x] `posembedding/qkv_rms_norm_rope_cache`
- [x] `posembedding/rope_quant_kvcache`
- [~] `posembedding/rope_with_sin_cos_cache`
- [x] `posembedding/rotary_position_embedding`
- [x] `posembedding/rotary_position_embedding_grad`
- [x] `gmm/grouped_matmul`
- [x] `gmm/grouped_matmul_add`
- [~] `gmm/grouped_matmul_finalize_routing`
- [~] `gmm/grouped_matmul_swiglu_quant`
- [~] `gmm/grouped_matmul_swiglu_quant_v2`
- [~] `gmm/quant_grouped_matmul_inplace_add`
- [~] `ffn/ffn`
- [~] `ffn/swin_attention_ffn`
- [x] `ffn/swin_transformer_ln_qkv`
- [~] `ffn/swin_transformer_ln_qkv_quant`

## Wave 2

- [~] `moe/moe_compute_expert_tokens`
- [~] `moe/moe_finalize_routing`
- [~] `moe/moe_finalize_routing_v2`
- [~] `moe/moe_finalize_routing_v2_grad`
- [~] `moe/moe_gating_top_k`
- [~] `moe/moe_gating_top_k_softmax`
- [~] `moe/moe_gating_top_k_softmax_v2`
- [~] `moe/moe_init_routing`
- [ ] `moe/moe_init_routing_quant`
- [ ] `moe/moe_init_routing_quant_v2`
- [~] `moe/moe_init_routing_v2`
- [ ] `moe/moe_init_routing_v2_grad`
- [ ] `moe/moe_init_routing_v3`
- [~] `moe/moe_re_routing`
- [x] `moe/moe_token_permute`
- [~] `moe/moe_token_permute_grad`
- [ ] `moe/moe_token_permute_with_ep`
- [ ] `moe/moe_token_permute_with_ep_grad`
- [x] `moe/moe_token_permute_with_routing_map`
- [~] `moe/moe_token_permute_with_routing_map_grad`
- [x] `moe/moe_token_unpermute`
- [~] `moe/moe_token_unpermute_grad`
- [ ] `moe/moe_token_unpermute_with_ep`
- [ ] `moe/moe_token_unpermute_with_ep_grad`
- [x] `moe/moe_token_unpermute_with_routing_map`
- [~] `moe/moe_token_unpermute_with_routing_map_grad`

## Wave 3

- [x] `attention/attention_update`
- [x] `attention/flash_attention_score`
- [ ] `attention/flash_attention_score_grad`
- [x] `attention/fused_infer_attention_score`
- [x] `attention/incre_flash_attention`
- [x] `attention/prompt_flash_attention`
- [~] `attention/recurrent_gated_delta_rule`
- [~] `attention/ring_attention_update`
- [ ] `attention/scatter_pa_cache`

## Wave 4

- [ ] `attention/gather_pa_kv_cache`
- [ ] `attention/kv_quant_sparse_flash_attention`
- [ ] `attention/lightning_indexer`
- [ ] `attention/mla_preprocess`
- [ ] `attention/mla_prolog`
- [ ] `attention/mla_prolog_v2`
- [ ] `attention/mla_prolog_v3`
- [ ] `attention/nsa_compress`
- [ ] `attention/nsa_compress_attention`
- [ ] `attention/nsa_compress_attention_infer`
- [ ] `attention/nsa_compress_grad`
- [ ] `attention/nsa_compress_with_cache`
- [ ] `attention/nsa_selected_attention`
- [ ] `attention/nsa_selected_attention_grad`
- [ ] `attention/nsa_selected_attention_infer`
- [ ] `attention/quant_lightning_indexer`
- [ ] `attention/sparse_flash_attention`

## Wave 5

- [~] `mc2/all_gather_matmul`
- [x] `mc2/grouped_mat_mul_all_reduce`
- [x] `mc2/inplace_matmul_all_reduce_add_rms_norm`
- [x] `mc2/matmul_all_reduce`
- [x] `mc2/matmul_all_reduce_add_rms_norm`
- [x] `mc2/matmul_reduce_scatter`
- [x] `mc2/moe_distribute_combine`
- [ ] `mc2/moe_distribute_combine_v2`
- [x] `mc2/moe_distribute_dispatch`
- [ ] `mc2/moe_distribute_dispatch_v2`
- [ ] `mc2/moe_distribute_dispatch_v3`
