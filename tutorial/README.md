# PTO Kernels Tutorial

面向初学者的 `pto-kernels` 教程，从整体架构、最小 kernel、真实算子，到性能和调试，按章节组织。

## 这套教程适合谁

- 会 Python，但还不熟悉 `PTO-DSL -> PTOAS -> PTO-ISA` 这一套链路的人
- 想从 `ops-transformer` 迁移一个 NPU kernel 到 PTO 的工程师
- 想读懂本仓库 `kernel.py / meta.py / report.json` 之间关系的人

## 读前准备

- 已完成 [README.md](../README.md) 里的环境初始化
- 能理解张量 shape、tile、block、softmax、matmul 这些基础概念
- 知道 `torch_npu` 是 baseline 运行时，但不要求熟悉 Ascend C

## 推荐阅读顺序

1. [01 架构总览](./01_architecture_overview.md)
2. [02 仓库结构与工作流](./02_repo_layout_and_workflow.md)
3. [03 第一个 PTO Kernel](./03_your_first_kernel.md)
4. [04 Matmul 与 Grouped Matmul](./04_matmul_and_grouped_matmul.md)
5. [05 Attention Kernel](./05_attention_kernels.md)
6. [06 MoE 与 Routing Kernel](./06_moe_and_routing.md)
7. [07 读懂 `.pto` 与调试](./07_reading_pto_and_debugging.md)
8. [08 性能解读](./08_performance_guide.md)
9. [09 新增一个 Kernel 的方法](./09_how_to_add_a_new_kernel.md)

## 你会得到什么

读完这套教程，应该能回答下面几个问题：

- `PTO-DSL`、`PTOAS`、`PTO-ISA` 分别负责什么
- 一个 PTO kernel 在本仓库里为什么不只是 `kernel.py`
- 为什么有些 kernel 是 `green + tile-first`，有些是 `scalar-heavy`
- 为什么有些 kernel `PTO ok` 但 baseline blocked
- 怎么从 Python 代码一路看到 `.pto`、`.cpp`、`.so` 和 benchmark 报告

## 仓库里的真实入口

这一套教程会反复用到这些真实路径：

- kernel source: `python/pto_kernels/ops/<family>/<op>/kernel.py`
- metadata: `python/pto_kernels/ops/<family>/<op>/meta.py`
- spec: `bench/specs/<family>/<op>.yaml`
- baseline adapter: `bench/adapters/ops_transformer/...`
- PTO adapter: `bench/adapters/ptodsl/...`
- latest report: `bench/generated/<family>/<op>/report.json`
- summary: `bench/reports/regression_latest.md`

## 常见误区

- 误区 1：`pto-kernels` 只是一个 kernel 代码仓库  
  不是。它同时承载了 benchmark、inventory、migration checklist、debug artifacts。
- 误区 2：写完 `kernel.py` 就算完成  
  不是。最少还要有 `meta.py`、spec、adapter、report。
- 误区 3：PTO-ISA 是给大多数用户直接写业务逻辑的 API  
  不是。大部分用户写的是 PTODSL；PTO-ISA 更像底层 tile 指令语义和 backend 合约。

下一章：从整体链路开始，把几个核心仓库和执行阶段放进同一张图里看。
