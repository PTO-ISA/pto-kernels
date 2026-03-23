# 09. 新增一个 Kernel 的方法

## 这一章你会学到什么

- 在 `pto-kernels` 里新增一个 kernel 需要补哪些文件
- 怎样把一个 kernel 放进 benchmark 和 migration 体系
- 怎样判断它目前属于 green、scalar-heavy，还是 blocker 状态

## 最小落地清单

新增一个 kernel，至少要补这些东西：

1. `python/pto_kernels/ops/<family>/<op>/kernel.py`
2. `python/pto_kernels/ops/<family>/<op>/meta.py`
3. `bench/specs/<family>/<op>.yaml`
4. `bench/adapters/ops_transformer/<family>/<op>.py`
5. `bench/adapters/ptodsl/<family>/<op>.py`
6. `bench/generated/<family>/<op>/report.json`

如果没有这些文件，你写的东西通常还不能算“进入迁移体系”。

## 推荐实操顺序

### 1. 先选 slice

不要一上来追求 full semantics，先选一个收敛 slice：

- 最小可对齐 shape
- 最少 feature 开关
- 最小 baseline contract

### 2. 先写 `kernel.py`

目标不是一次到最优，而是先跑通：

- 先选 `section.vector` 还是 `section.cube`
- 先做 tile-first 主路径
- 避免过早把复杂 host contract 全塞进 hot path

### 3. 再写 `meta.py`

在这里明确：

- 当前 slice 约束
- 当前 wave
- 当前 blocker
- 当前是否 scalar-heavy

### 4. 接上 spec 和 adapter

没有这一步，就没有系统性的 benchmark 和状态同步。

### 5. 生成并检查 report

至少要看：

- baseline 状态
- PTO 状态
- correctness
- latency
- blocked reason

## 一个简单 checklist

```text
[ ] kernel.py 能 build
[ ] meta.py 说明当前 slice 和 blocker
[ ] spec 已接入 benchmark
[ ] baseline adapter 可运行或明确 blocked
[ ] PTO adapter 可运行或明确 blocked
[ ] report.json 已生成
[ ] regression_latest.md 有对应行
[ ] kernel_state_matrix_latest.md 状态正确
```

## 如何给 kernel 分类

在当前仓库里，至少要区分：

- `green + tile-first`
- `green but scalar-heavy`
- `blocked by PTODSL surface`
- `blocked by PTOAS lowering`
- `blocked by pto-isa/backend capability`
- `blocked by host baseline/runtime gap`

如果不先分类，后面很容易做错方向：

- 明明是 host baseline 缺失，却去扩 PTODSL
- 明明是 scalar-heavy 问题，却只盯着 correctness

## 从哪里抄“正确姿势”

最值得参考的真实例子：

- matmul 主线：
  - `python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`
  - `python/pto_kernels/ops/gmm/grouped_matmul_add/kernel.py`
- attention 主线：
  - `python/pto_kernels/ops/attention/flash_attention_score/kernel.py`
  - `python/pto_kernels/ops/attention/prompt_flash_attention/kernel.py`
- MoE 主线：
  - `python/pto_kernels/ops/moe/moe_token_permute/kernel.py`
  - `python/pto_kernels/ops/moe/moe_finalize_routing/kernel.py`

## 常见误区

- 误区：先补文档，代码和报告以后再补  
  这在本仓库里通常会让状态失真。
- 误区：只要 `kernel.py` 编译过就可以宣称迁移完成  
  还远远不够，至少要进入 report 和状态矩阵。

## 最后一句经验

写新 kernel 时，最好一直问自己两个问题：

1. 这个 kernel 现在是“能跑”，还是“迁移完成”
2. 当前真正的 blocker 在 PTODSL、PTOAS、PTO-ISA，还是 host baseline

如果这两个问题回答不清，后面的优化和文档都容易跑偏。
