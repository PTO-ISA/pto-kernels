# 07. 读懂 `.pto` 与调试

## 这一章你会学到什么

- `.pto` 应该怎么读
- 怎样把 `kernel.py`、`kernel.pto`、`kernel.cpp`、`report.json` 串起来
- blocker 在本仓库里通常怎么归类

## 先记住一个核心原则

不要把 `.pto` 当成“要手写的代码”，而要把它当成：

- PTODSL 生成出来的中间表达
- 用来确认结构有没有按预期成型
- 用来定位 blocker 在哪一层

## 看 `.pto` 时先看什么

第一眼先看这几个东西：

1. 有没有 `pto.section.vector` / `pto.section.cube`
2. tile / tensor view 是不是长得对
3. 有没有预期的 `tload` / `tstore` / `matmul` / `matmul_acc`
4. 控制流是不是落在你预期的位置

一个示意片段可以像这样读：

```text
pto.section.cube {
  %tv_a = pto.make_tensor_view ...
  %sv_a = pto.slice_view %tv_a ...
  pto.tload %sv_a, %a_mat
  pto.tmov %a_mat, %a_tile
  pto.tmatmul %a_tile, %b_tile, %c_tile
  pto.tstore %c_tile, %sv_out
}
```

读法不是“逐字翻译”，而是：

- 这段是不是我在 Python 里写的 cube 主路径
- 中间有没有多出不该有的 scalar/control-flow

## 从 Python 到 report 的真实链路

在本仓库里，建议你总是按这个顺序看：

1. `python/pto_kernels/ops/<family>/<op>/kernel.py`
2. `python/pto_kernels/ops/<family>/<op>/meta.py`
3. `bench/specs/<family>/<op>.yaml`
4. `bench/generated/<family>/<op>/report.json`
5. `bench/reports/kernel_state_matrix_latest.md`

如果是编译或 backend 问题，再去看保留下来的 `.pto` / `.cpp` artifact。

## blocker 的几种典型分类

本仓库里最常见的 blocker 分类有：

### 1. PTODSL surface

说明你在 Python 里还表达不出来，或者表达出来的写法不对。

### 2. PTOAS lowering

说明 `.pto` 已经出来了，但继续 lowering 出问题。

### 3. pto-isa / backend capability

说明更底层的 runtime、backend contract、或者 arch-specific 能力还不闭合。

### 4. host baseline/runtime gap

说明 PTO 路径未必有问题，但 baseline 在当前 host 上没有 entrypoint 或 contract 还原不出来。

## 一个真实例子

看：

- `bench/reports/kernel_state_matrix_latest.md`
- `bench/generated/attention/recurrent_gated_delta_rule/report.json`

你会发现一个 kernel 可能：

- baseline 能跑
- PTO blocked
- blocker 被归到 `pto-isa/backend capability`

这就说明问题不在“Python 写法”，而在更下层。

## 常见误区

- 误区：看到 `.pto` 就说明问题一定不在 PTODSL  
  不一定。PTODSL 也可能生成了“不够理想但合法”的 IR。
- 误区：report 里只看 latency 就够  
  不够。还要看 baseline/PTO 状态、correctness、blocked reason。

下一章：把这些状态放回性能视角，学会正确阅读当前仓库里的数据。
