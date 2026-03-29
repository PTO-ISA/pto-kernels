# 06. MoE 与 Routing Kernel

## 这一章你会学到什么

- 为什么 MoE/routing kernel 往往比 matmul 更容易变成 scalar-heavy
- `moe_token_permute` 为什么是更好的第一入口
- `moe_finalize_routing` 和 `moe_re_routing` 分别代表什么

## 先抓住这类 kernel 的本质

这类 kernel 常见的问题不是“算得不够多”，而是：

- token 怎么重排
- routing index 怎么读
- gather / scatter 怎么写
- metadata 怎么和主数据一起搬

所以它们经常出现在 `section.vector()` 里。

## 最适合初学者的入口：`moe_token_permute`

真实路径：

- `python/pto_kernels/ops/moe/moe_token_permute/kernel.py`

它的好处是：

- 逻辑清晰
- 主要展示 gather / store 这类数据移动
- 当前状态是 `green + tile-first`

代表性代码：

```python
with pto.section.vector():
    ...
    sv_tokens = pto.slice_view(...)
    pto.load(sv_tokens, tb_tokens)
    for row_idx in range(row_start, row_end, c1):
        sv_gather = pto.slice_view(...)
        sv_out = pto.slice_view(...)
        pto.load(sv_gather, tb_gather)
        pto.gather(tb_tokens, tb_out, tb_gather)
        pto.store(tb_out, sv_out)
```

这已经足够说明：

- routing kernel 不一定复杂在数学上
- 复杂的是搬运和索引

## 为什么有些 MoE kernel 是 `scalar-heavy`

看状态矩阵你会发现很多 MoE kernel 仍然是 `green but scalar-heavy`，例如：

- `moe_finalize_routing`
- `moe_gating_top_k`
- `moe_gating_top_k_softmax`
- `moe_re_routing`

原因通常不是“写错了”，而是：

- 真实 host contract 就带很多小标量索引
- 迁移早期先保证 correctness，再逐步移除 hot-path scalar

## `moe_finalize_routing` 适合讲什么

真实路径：

- `python/pto_kernels/ops/moe/moe_finalize_routing/kernel.py`

它能讲清楚：

- vector path 上怎么读 `expanded_row_idx`
- 怎么把 `expanded + bias`、`scale`、`x1` 组合起来
- 为什么 routing kernel 容易有 `load_scalar(...)`

这也是一个很好的“结构已清楚，但还值得继续去 scalar 化”的例子。

## `moe_re_routing` 适合讲什么

`moe_re_routing` 的意义不只是“又一个 MoE kernel”，而是它展示了真实迁移过程里的 blocker closure：

- 先 blocked
- 再通过 PTODSL / lowering 调整跑通
- 最后进入 green，但可能仍然 scalar-heavy

也就是说，教程里不应该把 kernel migration 讲得像一次性写完，而要讲成：

```text
先跑通
  -> 再归类 blocker
  -> 再减少 scalar hot path
```

## 当前性能怎么读

当前稳定结果中，MoE 代表样本大致是：

- `moe_token_permute`: `31.2%..31.3%`
- `moe_finalize_routing`: `33.3%..34.3%`
- `moe_re_routing`: `28.7%..29.0%`

这些数字不是特别高，但教程里它们非常有价值，因为它们讲清楚了：

- gather/scatter
- routing metadata
- scalar-heavy 和 tile-first 的边界

## 常见误区

- 误区：MoE kernel 的关键是乘法  
  很多时候不是。真正难的是 routing metadata 和数据重排。
- 误区：只要 correctness 通过就不用管 scalar-heavy  
  不对。迁移体系里，green 和 green + tile-first 是两个层次。

下一章：开始读 `.pto` 和调试 artifact，把“编译链路”真正看见。
