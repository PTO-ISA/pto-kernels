# Kernel Parity Summary

Generated: `20260312T091542Z`
Rounds per kernel: `3`

| Kernel | Input Shape | Baseline ms | PTO ms | Baseline / PTO | Correctness | Latest |
| --- | --- | ---: | ---: | ---: | --- | --- |
| grouped_matmul | `{"output": [1, 128, 128], "weight": [128, 128], "weight_v5": [1, 128, 128], "x": [128, 128]}` | 0.1048 | 0.2600 | 40.3% | pass | [report](bench/generated/gmm/grouped_matmul/report.json) |
| grouped_matmul | `{"output": [1, 128, 256], "weight": [128, 256], "weight_v5": [1, 128, 256], "x": [128, 128]}` | 0.1077 | 0.2607 | 41.3% | pass | [report](bench/generated/gmm/grouped_matmul/report.json) |
| grouped_matmul | `{"output": [1, 64, 128], "weight": [64, 128], "weight_v5": [1, 64, 128], "x": [64, 64]}` | 0.1050 | 0.2588 | 40.6% | pass | [report](bench/generated/gmm/grouped_matmul/report.json) |
| apply_rotary_pos_emb | `{"cos": [64, 1, 128], "key": [64, 1, 128], "query": [64, 1, 128], "sin": [64, 1, 128]}` | 0.1014 | 0.2082 | 48.7% | pass | [report](bench/generated/posembedding/apply_rotary_pos_emb/report.json) |
| apply_rotary_pos_emb | `{"cos": [2, 32, 1, 128], "key": [2, 32, 1, 128], "query": [2, 32, 1, 128], "sin": [2, 32, 1, 128]}` | 0.1004 | 0.2071 | 48.5% | pass | [report](bench/generated/posembedding/apply_rotary_pos_emb/report.json) |
| ffn | `{"hidden": [32, 256], "output": [32, 128], "weight1": [128, 256], "weight2": [256, 128], "x": [32, 128]}` | 0.0896 | 0.4314 | 20.8% | pass | [report](bench/generated/ffn/ffn/report.json) |
| ffn | `{"hidden": [64, 256], "output": [64, 128], "weight1": [128, 256], "weight2": [256, 128], "x": [64, 128]}` | 0.0905 | 0.4289 | 21.1% | pass | [report](bench/generated/ffn/ffn/report.json) |
| ffn | `{"hidden": [32, 512], "output": [32, 128], "weight1": [128, 512], "weight2": [512, 128], "x": [32, 128]}` | 0.0960 | 0.4296 | 22.3% | pass | [report](bench/generated/ffn/ffn/report.json) |
| moe_token_permute | `{"gather_indices": [128], "indices": [8], "tokens": [8, 16]}` | 0.1035 | 0.3407 | 30.4% | pass | [report](bench/generated/moe/moe_token_permute/report.json) |
| moe_token_permute | `{"gather_indices": [256], "indices": [16], "tokens": [16, 16]}` | 0.1055 | 0.3438 | 30.7% | pass | [report](bench/generated/moe/moe_token_permute/report.json) |
| moe_token_permute | `{"gather_indices": [512], "indices": [16], "tokens": [16, 32]}` | 0.1048 | 0.3439 | 30.5% | pass | [report](bench/generated/moe/moe_token_permute/report.json) |
| flash_attention_score | `{"key": [1, 1, 32, 128], "output": [1, 1, 32, 128], "query": [1, 1, 32, 128], "scores": [32, 32], "value": [1, 1, 32, 128]}` | 0.1554 | 0.4353 | 35.7% | pass | [report](bench/generated/attention/flash_attention_score/report.json) |
| flash_attention_score | `{"key": [1, 1, 32, 64], "output": [1, 1, 32, 64], "query": [1, 1, 32, 64], "scores": [32, 32], "value": [1, 1, 32, 64]}` | 0.1547 | 0.4346 | 35.6% | pass | [report](bench/generated/attention/flash_attention_score/report.json) |
| flash_attention_score | `{"key": [1, 1, 64, 64], "output": [1, 1, 64, 64], "query": [1, 1, 64, 64], "scores": [64, 64], "value": [1, 1, 64, 64]}` | 0.1556 | 0.4377 | 35.6% | pass | [report](bench/generated/attention/flash_attention_score/report.json) |
| matmul_reduce_scatter | `{"local_mm": [128, 128], "reduce_scatter_output_per_rank": [64, 128], "world_size": 2, "x1": [128, 256], "x2": [256, 128]}` | 0.5681 | 0.7296 | 77.9% | pass | [report](bench/generated/mc2/matmul_reduce_scatter/report.json) |
| matmul_reduce_scatter | `{"local_mm": [64, 128], "reduce_scatter_output_per_rank": [32, 128], "world_size": 2, "x1": [64, 256], "x2": [256, 128]}` | 0.5316 | 0.7176 | 74.1% | pass | [report](bench/generated/mc2/matmul_reduce_scatter/report.json) |