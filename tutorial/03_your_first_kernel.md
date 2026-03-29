# 03. 第一个 PTO Kernel

## 这一章你会学到什么

- PTODSL kernel 的基本骨架是什么
- `section.vector` 和 `section.cube` 分别适合什么
- 怎么从 Python 心智模型过渡到 `.pto`

## 先看一个最小可读骨架

下面不是仓库里的整文件，而是最小可读版本：

```python
from ptodsl import jit, pto

const = pto.const

@jit(meta_data=..., output_dir=..., block_dim=8, enable_insert_sync=True, npu_arch="dav-2201")
def tiny_add(out_ptr: "ptr", x_ptr: "ptr", y_ptr: "ptr") -> None:
    c0 = const(0)
    c1 = const(1)
    cN = const(128)

    tv_x = pto.as_tensor(tensor, ptr=x_ptr, shape=[cN], strides=[c1])
    tv_y = pto.as_tensor(tensor, ptr=y_ptr, shape=[cN], strides=[c1])
    tv_out = pto.as_tensor(tensor, ptr=out_ptr, shape=[cN], strides=[c1])

    with pto.section.vector():
        x_tile = pto.alloc_tile(tile_vec)
        y_tile = pto.alloc_tile(tile_vec)
        out_tile = pto.alloc_tile(tile_vec)
        sv_x = pto.slice_view(sub_vec, source=tv_x, offsets=[c0], sizes=[cN])
        sv_y = pto.slice_view(sub_vec, source=tv_y, offsets=[c0], sizes=[cN])
        sv_out = pto.slice_view(sub_vec, source=tv_out, offsets=[c0], sizes=[cN])
        pto.load(sv_x, x_tile)
        pto.load(sv_y, y_tile)
        pto.add(x_tile, y_tile, out_tile)
        pto.store(out_tile, sv_out)
```

你应该先抓住的不是每个 API 名，而是这个模式：

```text
as_tensor -> alloc_tile -> slice_view -> load -> compute -> store
```

## `section.vector` 和 `section.cube`

### `section.vector`

适合：

- add
- mul
- softmax 中的 row-wise 向量处理
- routing / gather / scatter 类数据搬运

### `section.cube`

适合：

- matmul
- qk / pv 这类 cube-heavy 阶段
- 任何以 tile matmul 为主的块计算

## 一个示意性的 `.pto` 片段

下面是“你大概会看到什么”，不是要求你逐行背下来：

```text
pto.section.vector {
  %tv_x = pto.make_tensor_view ...
  %sv_x = pto.slice_view %tv_x ...
  pto.tload %sv_x, %x_tile
  pto.tload %sv_y, %y_tile
  pto.tadd %x_tile, %y_tile, %out_tile
  pto.tstore %out_tile, %sv_out
}
```

你可以把它理解成：

- Python 里写的是“逻辑描述”
- `.pto` 里看到的是“更接近 tile IR 的描述”

## 在本仓库里对应到哪里

可以先看这些真实文件：

- `python/pto_kernels/ops/moe/moe_token_permute/kernel.py`
- `python/pto_kernels/ops/attention/attention_update/kernel.py`
- `python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`

其中：

- `moe_token_permute` 更偏 vector
- `grouped_matmul` 更偏 cube

## 常见误区

- 误区：先把 API 全记下来再开始写  
  没必要。先记住 `as_tensor / slice_view / load / compute / store` 这条主线。
- 误区：`.pto` 应该和 Python 长得一模一样  
  不会。`.pto` 是更底层的 IR 表达，重点是结构能对上，而不是字面一一对应。

下一章：从最小骨架进入本仓库里最适合教学的 matmul 主线。
