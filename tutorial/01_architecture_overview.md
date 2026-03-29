# 01. 架构总览

## 这一章你会学到什么

- `PTO-DSL`、`PTOAS`、`PTO-ISA`、`Bisheng` 各自负责什么
- 一个 PTO kernel 从 Python 到 `.so` 的最短路径是什么
- baseline 和 PTO path 为什么要同时存在

## 一张文字版流程图

```text
kernel.py
  -> PTODSL build
  -> kernel.pto
  -> PTOAS
  -> kernel.cpp / backend lowering
  -> bisheng
  -> kernel.so
  -> PTO adapter
  -> report.json / regression_latest.md
```

同时，本仓库还维护一条 baseline 路径：

```text
ops-transformer adapter
  -> torch_npu / torch.ops.npu
  -> baseline output + latency
```

最后两条路径在 benchmark 框架里汇合，形成对比结果。

## 三层职责怎么区分

### PTODSL

这是大多数用户真正写代码的地方。

你主要会写这些东西：

```python
with pto.section.cube():
    tv_a = pto.as_tensor(...)
    sv_a = pto.slice_view(...)
    pto.load(sv_a, a_mat)
    pto.matmul(a_tile, b_tile, c_tile)
    pto.store(c_tile, sv_out)
```

它解决的是“如何描述 kernel 逻辑”。

### PTOAS

这是编译和 lowering 层。

它解决的是：

- `.pto` 是否合法
- section / tile / op 能不能继续往后端降
- 自动插 sync、规划内存、生成后端代码

如果一个 kernel 在这里失败，常见表现是：

- `PTOAS lowering blocked`
- `.pto` 生成了，但 `.cpp` 没生成成功
- 某个 `section.vector` / `section.cube` 里的控制流不被支持

### PTO-ISA

它更像 tile 级虚拟指令集和后端合约。

重点不是“用户天天直接写它”，而是：

- PTOAS 最终要对齐它的能力边界
- 某些 backend/blocker 最后会落到这里
- 它定义了 tile op、memory space、arch-specific 约束

一句更浅显的话：

- PTODSL 负责“写什么”
- PTOAS 负责“怎么编过去”
- PTO-ISA 负责“底层到底有哪些 tile 级能力”

## 在本仓库里对应到哪里

- PTODSL kernel 例子：
  - `python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`
  - `python/pto_kernels/ops/attention/flash_attention_score/kernel.py`
- benchmark 汇总：
  - `bench/reports/regression_latest.md`
- kernel 状态矩阵：
  - `bench/reports/kernel_state_matrix_latest.md`

## 为什么 baseline 很重要

本仓库不是只关心 PTO 能不能编过，更关心：

1. baseline 是否能跑
2. PTO 是否能跑
3. 两者是否对齐
4. 性能比例是多少

因此你会看到一些 kernel 状态是：

- `baseline ok / PTO ok`
- `baseline blocked / PTO ok`
- `baseline ok / PTO blocked`

这不是文档噪音，而是迁移工作的核心状态。

## 常见误区

- 误区：`.pto` 就是 PTO-ISA  
  不对。`.pto` 更接近 PTODSL 产出的 IR 文本，还要经过 PTOAS 和 backend。
- 误区：PTO kernel 跑通就等于迁移完成  
  不对。还要看 baseline 对齐、性能、block use、是否 scalar-heavy。

下一章：把抽象概念落到本仓库的目录和工作流上。
