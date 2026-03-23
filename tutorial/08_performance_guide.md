# 08. 性能解读

## 这一章你会学到什么

- `Baseline / PTO` 到底是什么意思
- `ok`、`blocked`、`pass` 应该怎么一起读
- 为什么 tutorial 里的“代表性 kernel”不等于“仓库里最快 kernel”

## 性能数据从哪里来

本教程统一引用：

- `bench/reports/regression_latest.md`
- `bench/reports/regression_latest.csv`
- `bench/reports/kernel_state_matrix_latest.md`

不手写、不猜测、不临时重新 benchmark。

## 最重要的几个字段

### `Baseline`

表示 baseline 路径是否稳定可跑。

### `PTO`

表示 PTO 路径是否稳定可跑。

### `Baseline / PTO`

表示 baseline latency 除以 PTO latency 的比例。

举例：

- `80%` 左右：PTO 已经比较接近 baseline
- `30%` 左右：PTO 能跑，但还有较大优化空间
- `n/a`：通常说明 baseline blocked 或 PTO blocked

## 代表性样本

### 高表现 tile-first

| Kernel | Ratio |
| --- | ---: |
| `grouped_mat_mul_all_reduce` | `89.4%..94.5%` |
| `prompt_flash_attention` | `71.0%..87.4%` |
| `matmul_reduce_scatter` | `76.6%..82.7%` |

这些 kernel 说明：

- PTO 路径已经非常接近业务可用形态
- tile-first 结构通常更利于持续优化

### 中位表现

| Kernel | Ratio |
| --- | ---: |
| `grouped_matmul` | `42.3%..42.5%` |
| `flash_attention_score` | `36.2%..37.3%` |
| `fused_infer_attention_score` | `33.0%..41.1%` |

这些 kernel 很适合教学，因为：

- 结构清晰
- baseline 和 PTO 都稳定
- 同时还能看到真实性能差距

### 明显慢但已跑通

| Kernel | Ratio |
| --- | ---: |
| `ffn` | `18.9%..20.7%` |
| `incre_flash_attention` | `24.0%..24.8%` |

这些例子提醒你：

> “能跑通”和“性能好”是两个不同层级的目标。

### baseline 外部阻塞

| Kernel | 状态 |
| --- | --- |
| `qkv_rms_norm_rope_cache` | baseline blocked, PTO ok (`0.7783 ms`) |
| `moe_finalize_routing_v2` | baseline blocked, PTO ok (`0.3181 ms`) |

这种状态不是失败，而是说明：

- PTO 已经写出来并跑通
- 但当前 host 上没有可用 baseline entrypoint

### backend/blocker 样本

| Kernel | 状态 |
| --- | --- |
| `recurrent_gated_delta_rule` | baseline ok, PTO blocked |

这个例子适合说明：

- 有时候 PTODSL 和 PTOAS 都不是主问题
- 问题可能最终落在 backend/runtime 合约

## 为什么有些 kernel 是 green 但仍然 scalar-heavy

看 `bench/reports/kernel_state_matrix_latest.md` 你会发现：

- `green + tile-first`
- `green but scalar-heavy`

这是两个不同状态。

区别在于：

- `green`：当前 correctness / benchmark 状态稳定
- `scalar-heavy`：hot path 里仍有较多 `load_scalar / store_scalar / scalar_select`

所以：

> 一个 kernel 可以“能用”，但还不算迁移质量最好。

## 为什么教程优先讲“结构正确、可解释”的 kernel

教程的目标不是罗列最快排行榜，而是帮读者建立正确心智模型。

因此会优先讲：

- `grouped_matmul`
- `grouped_matmul_add`
- `flash_attention_score`
- `prompt_flash_attention`
- `moe_token_permute`

而不是先从最难、最不稳定或最底层 blocker 最多的 kernel 开始。

## 常见误区

- 误区：比例低就说明 kernel 没价值  
  不对。很多低比例 kernel 仍然是很好的教学样本。
- 误区：`baseline blocked / PTO ok` 就没法讲性能  
  不对。仍然可以讲 PTO 自身性能、结构和 blocker 性质，只是不能做直接对比。

下一章：把前面内容收束成一个新增 kernel 的实操 checklist。
