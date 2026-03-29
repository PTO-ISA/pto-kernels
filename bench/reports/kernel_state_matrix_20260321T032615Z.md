# Kernel State Matrix

Generated: `20260321T032615Z`

| Kernel | Wave | Class | Scalar Hot Path | Block Use | Baseline | PTO | Baseline / PTO | Latest |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| moe_compute_expert_tokens | wave2 | blocked by PTOAS lowering | yes | n/a | ok | blocked | n/a | [report](bench/generated/moe/moe_compute_expert_tokens/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_compute_expert_tokens/kernel.py:load_scalarx1; python/pto_kernels/ops/moe/moe_compute_expert_tokens/kernel.py:store_scalarx1; python/pto_kernels/ops/moe/moe_compute_expert_tokens/kernel.py:scalar_selectx1` |  |  |  |  |  |  |
| swin_transformer_ln_qkv_quant | wave1 | blocked by PTODSL surface | no | n/a | blocked | blocked | n/a | [report](bench/generated/ffn/swin_transformer_ln_qkv_quant/report.json) |
|  |  | gap ids: `ptodsl-ffn-interstage-cast-path, ops-transformer-swin-ln-qkv-quant-python-entrypoint-gap, ptodsl-swin-ln-qkv-quant-surface` |  |  |  |  |  |  |
| grouped_matmul_finalize_routing | wave1 | blocked by PTODSL surface | no | n/a | blocked | blocked | n/a | [report](bench/generated/gmm/grouped_matmul_finalize_routing/report.json) |
|  |  | gap ids: `ptodsl-group-routing-primitives, ops-transformer-grouped-finalize-routing-format-contract` |  |  |  |  |  |  |
| ring_attention_update | wave3 | blocked by PTODSL surface | no | n/a | blocked | blocked | n/a | [report](bench/generated/attention/ring_attention_update/report.json) |
|  |  | gap ids: `host-ring-attention-update-python-entrypoint, ptodsl-ring-attention-update-list-surface` |  |  |  |  |  |  |
| recurrent_gated_delta_rule | wave3 | blocked by pto-isa/backend capability | no | n/a | ok | blocked | n/a | [report](bench/generated/attention/recurrent_gated_delta_rule/report.json) |
|  |  | gap ids: `ptoisa-a3-recurrent-state-update-runtime-hang` |  |  |  |  |  |  |
| moe_init_routing | wave2 | blocked by PTO correctness gap | no | n/a | ok | ok | 16.2%..16.7% | [report](bench/generated/moe/moe_init_routing/report.json) |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| swin_attention_ffn | wave1 | blocked by host baseline/runtime gap | yes | 24 blocks, all=True | blocked | ok | n/a | [report](bench/generated/ffn/swin_attention_ffn/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/ffn/common.py:scalar_selectx2` |  |  |  |  |  |  |
|  |  | gap ids: `ptoas-memory-plan-sync-pipeline, ptodsl-ffn-interstage-cast-path, ops-transformer-swin-ffn-python-entrypoint-gap, ptodsl-swin-layout-shift-surface` |  |  |  |  |  |  |
| swin_transformer_ln_qkv | wave1 | blocked by host baseline/runtime gap | no | 24 blocks, all=True | blocked | ok | n/a | [report](bench/generated/ffn/swin_transformer_ln_qkv/report.json) |
|  |  | gap ids: `ptodsl-ffn-interstage-cast-path, ops-transformer-swin-ln-qkv-python-entrypoint-gap` |  |  |  |  |  |  |
| grouped_matmul_swiglu_quant | wave1 | blocked by host baseline/runtime gap | no | n/a | blocked | blocked | n/a | [report](bench/generated/gmm/grouped_matmul_swiglu_quant/report.json) |
|  |  | gap ids: `ops-transformer-grouped-swiglu-quant-format-contract` |  |  |  |  |  |  |
| grouped_matmul_swiglu_quant_v2 | wave1 | blocked by host baseline/runtime gap | no | n/a | blocked | blocked | n/a | [report](bench/generated/gmm/grouped_matmul_swiglu_quant_v2/report.json) |
|  |  | gap ids: `ops-transformer-grouped-swiglu-quant-format-contract` |  |  |  |  |  |  |
| quant_grouped_matmul_inplace_add | wave1 | blocked by host baseline/runtime gap | no | n/a | blocked | blocked | n/a | [report](bench/generated/gmm/quant_grouped_matmul_inplace_add/report.json) |
|  |  | gap ids: `ops-transformer-quant-grouped-matmul-inplace-add-entrypoint-gap` |  |  |  |  |  |  |
| qkv_rms_norm_rope_cache | wave1 | blocked by host baseline/runtime gap | no | n/a | blocked | ok | n/a | [report](bench/generated/posembedding/qkv_rms_norm_rope_cache/report.json) |
|  |  | gap ids: `ptodsl-rope-generalization, ptodsl-vector-interleave-surface, ptodsl-cache-update-primitives, ptoisa-a2a3-vec-quant-store, ops-transformer-qkv-rms-norm-rope-cache-python-entrypoint-gap` |  |  |  |  |  |  |
| moe_finalize_routing_v2 | wave2 | blocked by host baseline/runtime gap | yes | 8 blocks, all=True | blocked | ok | n/a | [report](bench/generated/moe/moe_finalize_routing_v2/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_finalize_routing_v2/kernel.py:load_scalarx3` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_finalize_routing_v2_grad | wave2 | blocked by host baseline/runtime gap | yes | 8 blocks, all=True | blocked | ok | n/a | [report](bench/generated/moe/moe_finalize_routing_v2_grad/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_finalize_routing_v2_grad/kernel.py:load_scalarx3` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_gating_top_k_softmax_v2 | wave2 | blocked by host baseline/runtime gap | yes | 8 blocks, all=True | blocked | ok | n/a | [report](bench/generated/moe/moe_gating_top_k_softmax_v2/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_gating_top_k_softmax_v2/kernel.py:load_scalarx3; python/pto_kernels/ops/moe/moe_gating_top_k_softmax_v2/kernel.py:store_scalarx3; python/pto_kernels/ops/moe/moe_gating_top_k_softmax_v2/kernel.py:scalar_selectx2` |  |  |  |  |  |  |
|  |  | gap ids: `ops-transformer-moe-gating-top-k-softmax-v2-python-entrypoint-gap` |  |  |  |  |  |  |
| moe_init_routing_v2 | wave2 | blocked by host baseline/runtime gap | yes | n/a | blocked | ok | n/a | [report](bench/generated/moe/moe_init_routing_v2/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_init_routing_v2/kernel.py:load_scalarx3; python/pto_kernels/ops/moe/moe_init_routing_v2/kernel.py:store_scalarx4; python/pto_kernels/ops/moe/moe_init_routing_v2/kernel.py:scalar_selectx1` |  |  |  |  |  |  |
|  |  | gap ids: `ops-transformer-moe-init-routing-v2-host-crash` |  |  |  |  |  |  |
| moe_distribute_combine | wave5 | blocked by host baseline/runtime gap | no | n/a | blocked | ok | n/a | [report](bench/generated/mc2/moe_distribute_combine/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ptodsl-routing-sort-primitives, ops-transformer-mc2-hccl-multirank-bringup` |  |  |  |  |  |  |
| moe_distribute_dispatch | wave5 | blocked by host baseline/runtime gap | no | n/a | blocked | ok | n/a | [report](bench/generated/mc2/moe_distribute_dispatch/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ops-transformer-mc2-hccl-multirank-bringup` |  |  |  |  |  |  |
| ffn | wave1 | green but scalar-heavy | yes | n/a | ok | ok | 18.9%..20.7% | [report](bench/generated/ffn/ffn/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/ffn/common.py:scalar_selectx2` |  |  |  |  |  |  |
| moe_finalize_routing | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 33.3%..34.3% | [report](bench/generated/moe/moe_finalize_routing/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_finalize_routing/kernel.py:load_scalarx3` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_gating_top_k | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 54.1%..54.9% | [report](bench/generated/moe/moe_gating_top_k/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_gating_top_k/kernel.py:load_scalarx2; python/pto_kernels/ops/moe/moe_gating_top_k/kernel.py:store_scalarx2; python/pto_kernels/ops/moe/moe_gating_top_k/kernel.py:scalar_selectx2` |  |  |  |  |  |  |
| moe_gating_top_k_softmax | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 26.4%..26.8% | [report](bench/generated/moe/moe_gating_top_k_softmax/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_gating_top_k_softmax/kernel.py:load_scalarx3; python/pto_kernels/ops/moe/moe_gating_top_k_softmax/kernel.py:store_scalarx3; python/pto_kernels/ops/moe/moe_gating_top_k_softmax/kernel.py:scalar_selectx2` |  |  |  |  |  |  |
| moe_re_routing | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 28.7%..29.0% | [report](bench/generated/moe/moe_re_routing/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_re_routing/kernel.py:load_scalarx2; python/pto_kernels/ops/moe/moe_re_routing/kernel.py:store_scalarx3; python/pto_kernels/ops/moe/moe_re_routing/kernel.py:scalar_selectx6` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-group-routing-primitives` |  |  |  |  |  |  |
| moe_token_permute_grad | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 43.1%..43.2% | [report](bench/generated/moe/moe_token_permute_grad/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_token_permute_grad/kernel.py:load_scalarx1` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_token_permute_with_routing_map_grad | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 31.2%..32.0% | [report](bench/generated/moe/moe_token_permute_with_routing_map_grad/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_token_permute_with_routing_map_grad/kernel.py:load_scalarx1; python/pto_kernels/ops/moe/moe_token_permute_with_routing_map_grad/kernel.py:store_scalarx1` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_token_unpermute_grad | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 37.7%..40.8% | [report](bench/generated/moe/moe_token_unpermute_grad/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_token_unpermute_grad/kernel.py:load_scalarx1; python/pto_kernels/ops/moe/moe_token_unpermute_grad/kernel.py:store_scalarx1` |  |  |  |  |  |  |
| moe_token_unpermute_with_routing_map_grad | wave2 | green but scalar-heavy | yes | n/a | ok | ok | 41.2%..41.5% | [report](bench/generated/moe/moe_token_unpermute_with_routing_map_grad/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/moe/moe_token_unpermute_with_routing_map_grad/kernel.py:load_scalarx1; python/pto_kernels/ops/moe/moe_token_unpermute_with_routing_map_grad/kernel.py:store_scalarx1` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| all_gather_matmul | wave5 | green but scalar-heavy | yes | n/a | ok | ok | 58.4%..59.4% | [report](bench/generated/mc2/all_gather_matmul/report.json) |
|  |  | scalar matches: `python/pto_kernels/ops/mc2/all_gather_matmul/kernel.py:scalar_selectx2` |  |  |  |  |  |  |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ptodsl-cube-preload-pipeline-surface, ptoas-memory-plan-sync-pipeline, ops-transformer-mc2-hccl-multirank-bringup` |  |  |  |  |  |  |
| grouped_matmul | wave1 | green + tile-first | no | n/a | ok | ok | 42.3%..42.5% | [report](bench/generated/gmm/grouped_matmul/report.json) |
|  |  | gap ids: `ptodsl-cube-preload-pipeline-surface, ptodsl-group-routing-primitives` |  |  |  |  |  |  |
| grouped_matmul_add | wave1 | green + tile-first | no | n/a | ok | ok | 62.8%..67.4% | [report](bench/generated/gmm/grouped_matmul_add/report.json) |
|  |  | gap ids: `ptodsl-cube-preload-pipeline-surface` |  |  |  |  |  |  |
| apply_rotary_pos_emb | wave1 | green + tile-first | no | n/a | ok | ok | 46.5%..46.8% | [report](bench/generated/posembedding/apply_rotary_pos_emb/report.json) |
|  |  | gap ids: `ptodsl-vector-queue-pipeline-surface` |  |  |  |  |  |  |
| dequant_rope_quant_kvcache | wave1 | green + tile-first | no | n/a | ok | ok | 17.0%..18.0% | [report](bench/generated/posembedding/dequant_rope_quant_kvcache/report.json) |
|  |  | gap ids: `ptodsl-cache-update-primitives, ptoisa-a2a3-vec-quant-store` |  |  |  |  |  |  |
| interleave_rope | wave1 | green + tile-first | no | n/a | ok | ok | 29.1%..29.2% | [report](bench/generated/posembedding/interleave_rope/report.json) |
|  |  | gap ids: `ptodsl-vector-queue-pipeline-surface, ptodsl-vector-interleave-surface` |  |  |  |  |  |  |
| rope_quant_kvcache | wave1 | green + tile-first | no | n/a | ok | ok | 17.9%..18.0% | [report](bench/generated/posembedding/rope_quant_kvcache/report.json) |
|  |  | gap ids: `ptodsl-cache-update-primitives, ptoisa-a2a3-vec-quant-store` |  |  |  |  |  |  |
| rotary_position_embedding | wave1 | green + tile-first | no | n/a | ok | ok | 52.7%..53.5% | [report](bench/generated/posembedding/rotary_position_embedding/report.json) |
|  |  | gap ids: `ptodsl-rope-generalization, ptodsl-vector-queue-pipeline-surface` |  |  |  |  |  |  |
| rotary_position_embedding_grad | wave1 | green + tile-first | no | n/a | ok | ok | 39.5%..43.5% | [report](bench/generated/posembedding/rotary_position_embedding_grad/report.json) |
|  |  | gap ids: `ptodsl-rope-generalization, ptodsl-vector-queue-pipeline-surface` |  |  |  |  |  |  |
| moe_token_permute | wave2 | green + tile-first | no | n/a | ok | ok | 31.2%..31.3% | [report](bench/generated/moe/moe_token_permute/report.json) |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_token_permute_with_routing_map | wave2 | green + tile-first | no | n/a | ok | ok | 28.1%..29.0% | [report](bench/generated/moe/moe_token_permute_with_routing_map/report.json) |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_token_unpermute | wave2 | green + tile-first | no | n/a | ok | ok | 46.3%..51.6% | [report](bench/generated/moe/moe_token_unpermute/report.json) |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| moe_token_unpermute_with_routing_map | wave2 | green + tile-first | no | n/a | ok | ok | 44.0%..44.4% | [report](bench/generated/moe/moe_token_unpermute_with_routing_map/report.json) |
|  |  | gap ids: `ptodsl-routing-sort-primitives` |  |  |  |  |  |  |
| attention_update | wave3 | green + tile-first | no | n/a | ok | ok | 35.1%..36.0% | [report](bench/generated/attention/attention_update/report.json) |
| flash_attention_score | wave3 | green + tile-first | no | n/a | ok | ok | 36.2%..37.3% | [report](bench/generated/attention/flash_attention_score/report.json) |
| fused_infer_attention_score | wave3 | green + tile-first | no | n/a | ok | ok | 33.0%..41.1% | [report](bench/generated/attention/fused_infer_attention_score/report.json) |
| incre_flash_attention | wave3 | green + tile-first | no | n/a | ok | ok | 24.0%..24.8% | [report](bench/generated/attention/incre_flash_attention/report.json) |
| prompt_flash_attention | wave3 | green + tile-first | no | n/a | ok | ok | 71.0%..87.4% | [report](bench/generated/attention/prompt_flash_attention/report.json) |
| grouped_mat_mul_all_reduce | wave5 | green + tile-first | no | n/a | ok | ok | 89.4%..94.5% | [report](bench/generated/mc2/grouped_mat_mul_all_reduce/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ops-transformer-mc2-hccl-multirank-bringup` |  |  |  |  |  |  |
| inplace_matmul_all_reduce_add_rms_norm | wave5 | green + tile-first | no | n/a | ok | ok | 53.1%..59.5% | [report](bench/generated/mc2/inplace_matmul_all_reduce_add_rms_norm/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ptodsl-cube-preload-pipeline-surface, ptodsl-vector-rms-norm-generalization, ptoisa-a2a3-vector-cast-io, ptodsl-inplace-buffer-alias-contract` |  |  |  |  |  |  |
| matmul_all_reduce | wave5 | green + tile-first | no | n/a | ok | ok | 64.5%..75.7% | [report](bench/generated/mc2/matmul_all_reduce/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ptodsl-cube-preload-pipeline-surface, ops-transformer-mc2-hccl-multirank-bringup` |  |  |  |  |  |  |
| matmul_all_reduce_add_rms_norm | wave5 | green + tile-first | no | n/a | ok | ok | 63.6%..64.0% | [report](bench/generated/mc2/matmul_all_reduce_add_rms_norm/report.json) |
|  |  | gap ids: `ptodsl-multicore-collective-primitives, ptodsl-cube-preload-pipeline-surface, ptodsl-vector-rms-norm-generalization, ptoisa-a2a3-vector-cast-io` |  |  |  |  |  |  |
| matmul_reduce_scatter | wave5 | green + tile-first | no | n/a | ok | ok | 76.6%..82.7% | [report](bench/generated/mc2/matmul_reduce_scatter/report.json) |