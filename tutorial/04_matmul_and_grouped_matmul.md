# 04. Matmul 与 Grouped Matmul

## 这一章你会学到什么

- 为什么 `grouped_matmul` 是最好的 PTO 入门主线之一
- tile、block、swizzle 在真实 kernel 里怎么落地
- `grouped_matmul_add` 为什么先采用两阶段 epilogue

## 为什么先看 grouped matmul

当前仓库里，`grouped_matmul` 和 `grouped_matmul_add` 有几个优点：

- 都是 `green + tile-first`
- baseline 和 PTO 都稳定
- 结构清晰，能把 cube 主路径讲明白

真实路径：

- `python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`
- `python/pto_kernels/ops/gmm/grouped_matmul_add/kernel.py`

## 先看 `grouped_matmul` 的核心心智模型

最重要的不是记参数，而是理解它在做什么：

```text
把输出切成 baseM x baseN tiles
  -> 把 tiles 分给各个 block
  -> 每个 tile 内按 K 维做多次 matmul_acc
  -> 最后把 ACC tile 写回 GM
```

代码里最有代表性的一段是：

```python
for logical_block in range(bid, cTotalTiles, num_blocks):
    m_idx = logical_block // cNTiles
    n_idx = logical_block % cNTiles
    ...
    for i in range(c0, cIter, c1):
        k_off = i * cBaseK
        sv_a = pto.slice_view(...)
        sv_b = pto.slice_view(...)
        pto.load(sv_a, a_mat)
        pto.load(sv_b, b_mat)
        pto.mov(a_mat, a_tile)
        pto.mov(b_mat, b_tile)
        if i == c0:
            pto.matmul(a_tile, b_tile, c_tile)
        else:
            pto.matmul_acc(c_tile, a_tile, b_tile, c_tile)
```

这段就是 PTO 版 matmul 主路径的典型形态。

## tile / block / swizzle 分别在解决什么

### tile

tile 解决的是“单次算多大”。

本 kernel 用的代表性参数是：

- `base_m`
- `base_n`
- `base_k`

它们决定：

- `A_tile`
- `B_tile`
- `C_tile`

### block

block 解决的是“不同核心怎么分工”。

这里的 persistent-kernel 写法是：

```python
bid = pto.index_cast(pto.get_block_idx())
num_blocks = pto.index_cast(pto.get_block_num())
for logical_block in range(bid, cTotalTiles, num_blocks):
    ...
```

### swizzle

swizzle 解决的是“tile 遍历顺序怎么更友好”。

在本仓库里，grouped matmul 复用了共享 helper：

- `python/pto_kernels/ops/gmm/common.py`

这说明一个重要工程原则：

> swizzle 改的是 tile traversal，不改数学语义。

## 为什么 `grouped_matmul_add` 先做成两阶段

`grouped_matmul_add` 没有一开始就把 add 硬塞进 cube 主路径，而是拆成：

1. `stage_matmul`
2. `stage_add`

这在教学上反而更好，因为能清晰地区分：

- cube 负责什么
- vector epilogue 负责什么

`stage_add` 的代表性逻辑非常直观：

```python
for row_idx in range(row_start, row_end, c1):
    pto.load(sv_mm, mm_row)
    pto.load(sv_y, y_row)
    pto.add(mm_row, y_row, out_row)
    pto.store(out_row, sv_out)
```

## 当前性能怎么看

来自 `bench/reports/regression_latest.md` 的当前稳定结果：

- `grouped_matmul`: `42.3%..42.5%`
- `grouped_matmul_add`: `62.8%..67.4%`

这两个数字不是“越高越代表教程更重要”，而是说明：

- 结构清晰
- 对齐稳定
- 适合讲清 PTO kernel 的第一性原理

## 常见误区

- 误区：Grouped Matmul 就是多套 matmul 简单 for-loop 拼接  
  不够准确。真正重要的是 tile/block/schedule 怎么随着 group 维扩展。
- 误区：性能优化就是改几个 tile 参数  
  不够。tile 只是第一层，后面还有 swizzle、pipeline、epilogue、routing 合约。

下一章：进入 attention，看看为什么同样是 tile-first，attention 会明显更复杂。
