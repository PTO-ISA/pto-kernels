# Regression Parity Summary

Generated: `20260312T172408Z`
Rounds per kernel: `1`

| Kernel | Input Shape | Baseline ms | PTO ms | Baseline / PTO | Correctness | Latest |
| --- | --- | ---: | ---: | ---: | --- | --- |
| grouped_matmul | `{"output": [1, 128, 128], "weight": [128, 128], "weight_v5": [1, 128, 128], "x": [128, 128]}` | 0.1087 | 0.2612 | 41.6% | fail | [report](bench/generated/gmm/grouped_matmul/report.json) |
| grouped_matmul | `{"output": [1, 128, 256], "weight": [128, 256], "weight_v5": [1, 128, 256], "x": [128, 128]}` | 0.1096 | 0.2633 | 41.6% | fail | [report](bench/generated/gmm/grouped_matmul/report.json) |
| grouped_matmul | `{"output": [1, 64, 128], "weight": [64, 128], "weight_v5": [1, 64, 128], "x": [64, 64]}` | 0.1101 | 0.2621 | 42.0% | fail | [report](bench/generated/gmm/grouped_matmul/report.json) |
| apply_rotary_pos_emb | `{"cos": [64, 1, 128], "key": [64, 1, 128], "query": [64, 1, 128], "sin": [64, 1, 128]}` | 0.0993 | 0.2083 | 47.7% | fail | [report](bench/generated/posembedding/apply_rotary_pos_emb/report.json) |
| apply_rotary_pos_emb | `{"cos": [2, 32, 1, 128], "key": [2, 32, 1, 128], "query": [2, 32, 1, 128], "sin": [2, 32, 1, 128]}` | 0.1003 | 0.2074 | 48.3% | fail | [report](bench/generated/posembedding/apply_rotary_pos_emb/report.json) |
| ffn | `{"hidden": [32, 256], "output": [32, 128], "weight1": [128, 256], "weight2": [256, 128], "x": [32, 128]}` | 0.0875 | 0.4496 | 19.5% | fail | [report](bench/generated/ffn/ffn/report.json) |
| ffn | `{"hidden": [64, 256], "output": [64, 128], "weight1": [128, 256], "weight2": [256, 128], "x": [64, 128]}` | 0.0895 | 0.4541 | 19.7% | fail | [report](bench/generated/ffn/ffn/report.json) |
| ffn | `{"hidden": [32, 512], "output": [32, 128], "weight1": [128, 512], "weight2": [512, 128], "x": [32, 128]}` | 0.0953 | 0.4634 | 20.6% | fail | [report](bench/generated/ffn/ffn/report.json) |
| moe_token_permute | `{"gather_indices": [128], "indices": [8], "tokens": [8, 16]}` | 0.1076 | 0.3523 | 30.5% | fail | [report](bench/generated/moe/moe_token_permute/report.json) |
| moe_token_permute | `{"gather_indices": [256], "indices": [16], "tokens": [16, 16]}` | 0.1074 | 0.3524 | 30.5% | fail | [report](bench/generated/moe/moe_token_permute/report.json) |
| moe_token_permute | `{"gather_indices": [512], "indices": [16], "tokens": [16, 32]}` | 0.1084 | 0.3521 | 30.8% | fail | [report](bench/generated/moe/moe_token_permute/report.json) |
| flash_attention_score | `{"key": [1, 1, 32, 128], "output": [1, 1, 32, 128], "query": [1, 1, 32, 128], "scores": [32, 32], "value": [1, 1, 32, 128]}` | 0.1576 | 0.4420 | 35.7% | fail | [report](bench/generated/attention/flash_attention_score/report.json) |
| flash_attention_score | `{"key": [1, 1, 32, 64], "output": [1, 1, 32, 64], "query": [1, 1, 32, 64], "scores": [32, 32], "value": [1, 1, 32, 64]}` | 0.1629 | 0.4501 | 36.2% | fail | [report](bench/generated/attention/flash_attention_score/report.json) |
| flash_attention_score | `{"key": [1, 1, 64, 64], "output": [1, 1, 64, 64], "query": [1, 1, 64, 64], "scores": [64, 64], "value": [1, 1, 64, 64]}` | 0.1634 | 0.4443 | 36.8% | fail | [report](bench/generated/attention/flash_attention_score/report.json) |
| matmul_reduce_scatter | `{"local_mm": [128, 128], "reduce_scatter_output_per_rank": [64, 128], "world_size": 2, "x1": [128, 256], "x2": [256, 128]}` | 0.5996 | 0.7488 | 80.1% | fail | [report](bench/generated/mc2/matmul_reduce_scatter/report.json) |
| matmul_reduce_scatter | `{"local_mm": [64, 128], "reduce_scatter_output_per_rank": [32, 128], "world_size": 2, "x1": [64, 256], "x2": [256, 128]}` | 0.5246 | 0.7559 | 69.4% | fail | [report](bench/generated/mc2/matmul_reduce_scatter/report.json) |
| all_gather_matmul | `{"gather_out": [128, 256], "output": [128, 128], "world_size": 2, "x1_local": [64, 256], "x2": [256, 128]}` | 0.5492 | 0.9633 | 57.0% | fail | [report](bench/generated/mc2/all_gather_matmul/report.json) |
| all_gather_matmul | `{"gather_out": [256, 256], "output": [256, 128], "world_size": 2, "x1_local": [128, 256], "x2": [256, 128]}` | 0.5597 | 0.9757 | 57.4% | fail | [report](bench/generated/mc2/all_gather_matmul/report.json) |
| grouped_mat_mul_all_reduce | `{"group_list": [128], "groups": 1, "output": [128, 128], "weight_local": [128, 128], "world_size": 2, "x_local": [128, 128]}` | 0.6199 | 0.8192 | 75.7% | fail | [report](bench/generated/mc2/grouped_mat_mul_all_reduce/report.json) |
| grouped_mat_mul_all_reduce | `{"group_list": [256], "groups": 1, "output": [256, 128], "weight_local": [128, 128], "world_size": 2, "x_local": [256, 128]}` | 0.6979 | 0.7085 | 98.5% | fail | [report](bench/generated/mc2/grouped_mat_mul_all_reduce/report.json) |
| matmul_all_reduce | `{"output": [128, 128], "world_size": 2, "x1_local": [128, 256], "x2": [256, 128]}` | 0.6054 | 0.7039 | 86.0% | fail | [report](bench/generated/mc2/matmul_all_reduce/report.json) |
| matmul_all_reduce | `{"output": [256, 128], "world_size": 2, "x1_local": [256, 256], "x2": [256, 128]}` | 0.5382 | 0.7088 | 75.9% | fail | [report](bench/generated/mc2/matmul_all_reduce/report.json) |
| matmul_all_reduce_add_rms_norm | `{"gamma": [128], "norm_out": [128, 128], "residual": [128, 128], "world_size": 2, "x1_local": [128, 256], "x2": [256, 128], "y": [128, 128]}` | 0.6204 | 0.8665 | 71.6% | fail | [report](bench/generated/mc2/matmul_all_reduce_add_rms_norm/report.json) |
| matmul_all_reduce_add_rms_norm | `{"gamma": [128], "norm_out": [256, 128], "residual": [256, 128], "world_size": 2, "x1_local": [256, 256], "x2": [256, 128], "y": [256, 128]}` | 0.5624 | 0.9665 | 58.2% | fail | [report](bench/generated/mc2/matmul_all_reduce_add_rms_norm/report.json) |
| inplace_matmul_all_reduce_add_rms_norm | `{"gamma": [128], "norm_out": [128, 128], "residual_inout": [128, 128], "world_size": 2, "x1_local": [128, 256], "x2": [256, 128]}` | 0.5536 | 1.0519 | 52.6% | fail | [report](bench/generated/mc2/inplace_matmul_all_reduce_add_rms_norm/report.json) |
| inplace_matmul_all_reduce_add_rms_norm | `{"gamma": [128], "norm_out": [256, 128], "residual_inout": [256, 128], "world_size": 2, "x1_local": [256, 256], "x2": [256, 128]}` | 0.6747 | 1.1376 | 59.3% | fail | [report](bench/generated/mc2/inplace_matmul_all_reduce_add_rms_norm/report.json) |
| moe_distribute_dispatch | `n/a` | n/a | 27.9053 | n/a | fail | [report](bench/generated/mc2/moe_distribute_dispatch/report.json) |
| moe_distribute_combine | `n/a` | n/a | n/a | n/a | fail | [report](bench/generated/mc2/moe_distribute_combine/report.json) |