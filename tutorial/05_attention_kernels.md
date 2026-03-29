# 05. Attention Kernel

## 这一章你会学到什么

- attention 为什么比普通 matmul 更难
- `flash_attention_score`、`prompt_flash_attention`、`fused_infer_attention_score` 各自适合讲什么
- shared dense-attention path 是什么

## 为什么 attention 更难

matmul 主路径通常是：

```text
load -> matmul -> matmul_acc -> store
```

attention 至少会多出：

```text
QK -> softmax -> PV
```

也就是说除了 cube 计算，还有：

- row-wise softmax
- 分阶段中间结果
- 更复杂的 shape 约束

## 本仓库里的三个代表

### `flash_attention_score`

- 路径：`python/pto_kernels/ops/attention/flash_attention_score/kernel.py`
- 适合讲“最经典的 staged attention”
- 当前是 `green + tile-first`

### `prompt_flash_attention`

- 路径：`python/pto_kernels/ops/attention/prompt_flash_attention/kernel.py`
- 适合讲“在真实 host baseline 下已经比较接近的 attention kernel”
- 当前是 attention 家族里性能最好的稳定样本之一

### `fused_infer_attention_score`

- 路径：`python/pto_kernels/ops/attention/fused_infer_attention_score/kernel.py`
- 适合讲“为什么 infer contract 会比普通 dense attention 更容易出 correctness 细节问题”

## 共享的 dense-attention path

`flash_attention_score/kernel.py` 本身很短，因为真正的 staged builder 被抽到了公共模块里：

```python
from pto_kernels.ops.attention.common import DenseAttentionConfig, DenseAttentionPipelineWrapper

def build_jit_wrapper(*, output_dir):
    return DenseAttentionPipelineWrapper(config=_config(), output_dir=output_dir)
```

这说明一个很关键的工程实践：

> 真正复杂的 attention kernel，不应该把所有逻辑都塞进单文件。  
> 应该尽量把可复用的 QK / softmax / PV 路径抽出来。

## softmax 为什么值得特别关注

attention 的很多正确性问题，不在 matmul，而在 softmax：

- 数值稳定性
- row-wise reduction
- block_dim 对不同 shape 的影响

这也是为什么教程里会把 `fused_infer_attention_score` 作为例子：

- 它最终跑通了
- 但过程中 softmax 路径就出现过真实 correctness 调整

## 当前性能样本

来自当前稳定汇总：

- `flash_attention_score`: `36.2%..37.3%`
- `fused_infer_attention_score`: `33.0%..41.1%`
- `prompt_flash_attention`: `71.0%..87.4%`
- `incre_flash_attention`: `24.0%..24.8%`

从教程角度更重要的结论是：

- 同样叫 attention，不同 contract 的性能和难度差异非常大
- 不要把一个 green 的 attention kernel 当成“所有 attention 都差不多”

## 在仓库里怎么跟踪 attention

你最值得一起看的文件是：

- `python/pto_kernels/ops/attention/common.py`
- `bench/generated/attention/flash_attention_score/report.json`
- `bench/generated/attention/prompt_flash_attention/report.json`
- `bench/reports/kernel_state_matrix_latest.md`

## 常见误区

- 误区：attention 只是两次 matmul 加一个 softmax  
  形式上像，但工程复杂度明显更高，尤其是数值稳定和 contract 细节。
- 误区：只要 flash attention 绿了，infer attention 应该也差不多  
  不对。infer contract、cache shape、block 策略都会改变行为。

下一章：进入 MoE 和 routing 类 kernel，这类 kernel 里“数据搬运”往往比“乘加”更主导。
