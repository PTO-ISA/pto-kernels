# 02. 仓库结构与工作流

## 这一章你会学到什么

- `pto-kernels` 里哪些目录最重要
- 一个 kernel 为什么至少要有 5 类文件
- `kernel.py -> report.json` 的完整落地链路

## 最重要的几个目录

```text
python/pto_kernels/ops/      # PTO kernel 源码
bench/specs/                 # benchmark spec
bench/adapters/              # baseline/PTO 适配层
bench/generated/             # 每个 kernel 的最新稳定报告
bench/reports/               # 汇总报告和状态矩阵
docs/                        # bring-up / migration 文档
examples/                    # 示例代码
```

## 一个 kernel 的最小文件集合

以 `grouped_matmul` 为例，最关键的是：

- `python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`
- `python/pto_kernels/ops/gmm/grouped_matmul/meta.py`
- `bench/specs/gmm/grouped_matmul.yaml`
- `bench/adapters/ops_transformer/gmm/grouped_matmul.py`
- `bench/adapters/ptodsl/gmm/grouped_matmul.py`
- `bench/generated/gmm/grouped_matmul/report.json`

这 6 个路径说明了一个事实：

> 在本仓库里，kernel 不只是“写一个算子”，而是“把算子放进可 benchmark、可追踪、可汇总的迁移框架里”。

## 一个 kernel 的生命周期

### 1. 写 PTODSL kernel

在 `kernel.py` 里描述 tile、section、load/store、matmul、vector op。

### 2. 写元数据

在 `meta.py` 里记录：

- 当前 slice 的约束
- 状态
- 已知 blocker
- 是否 scalar-heavy

### 3. 写 benchmark spec

在 `bench/specs/...yaml` 里告诉 benchmark 框架：

- baseline adapter 是谁
- PTO adapter 是谁
- 正确性阈值
- 当前 wave / family / inventory 归属

### 4. 跑 adapter

- baseline adapter 调 `torch_npu` 或 `torch.ops.npu`
- PTO adapter 调用 PTODSL build 出来的 wrapper

### 5. 生成 report

最后产出：

- `bench/generated/<family>/<op>/report.json`
- `bench/reports/regression_latest.md`
- `bench/reports/kernel_state_matrix_latest.md`

## 一个真实链路

以 `flash_attention_score` 为例：

```text
python/pto_kernels/ops/attention/flash_attention_score/kernel.py
  -> build wrapper
  -> 生成 PTO IR / backend artifact
  -> bench/adapters/ptodsl/attention/flash_attention_score.py
  -> bench/generated/attention/flash_attention_score/report.json
  -> bench/reports/regression_latest.md
```

## 为什么 `meta.py` 很重要

初学者最容易忽略的是 `meta.py`。

但在这个仓库里，`meta.py` 不是可有可无的注释文件。它承载的是“迁移状态”：

- 这个 kernel 是不是 green
- 是不是 tile-first
- 有哪些 blocker
- 为什么还没合格

如果你只看 `kernel.py`，往往不知道它为什么没进 regression 绿表。

## 常见误区

- 误区：先把 kernel 写出来，其他文件以后补  
  在这个仓库里不成立。没有 spec/adapter/report，基本等于没有进入迁移体系。
- 误区：`bench/generated` 是临时目录，不重要  
  不对。这里是教程、汇总和状态同步的事实来源之一。

下一章：先不看复杂真实 kernel，先学会读一个最小 PTO kernel 长什么样。
