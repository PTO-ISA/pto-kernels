# Phase 1 Seed Audit

| Seed | Baseline | PTO | Baseline ms | PTO ms | PTO max abs diff |
| --- | --- | --- | ---: | ---: | ---: |
| posembedding/apply_rotary_pos_emb | ok/pass | ok/pass | 0.1066 | 0.2109 | 0.0 |
| gmm/grouped_matmul | ok/pass | ok/pass | 0.1085 | 0.2434 | 0.00048828125 |
| ffn/ffn | ok/pass | ok/pass | 0.0836 | 0.4334 | 0.0 |
| moe/moe_token_permute | ok/pass | ok/pass | 0.1074 | 0.3416 | 0.0 |
| attention/flash_attention_score | ok/pass | ok/pass | 0.1496 | 0.4295 | 6.103515625e-05 |
| mc2/matmul_reduce_scatter | ok/pass | ok/pass | 0.5391 | 0.7030 | 0.0008401870727539062 |
