# Quantized GMM Contract Probe

## grouped_matmul_finalize_routing
- status: `blocked`
- reason: Host baseline remains unstable: routed quantized tensors are required; ND int8 weights are rejected; supported W4A8-sized ND probes can segfault; NZ-format probing depends on working TBE Python deps.
- entrypoint: `torch_npu.npu_grouped_matmul_finalize_routing`
- probe_shapes: `[{"weight": [1, 128, 128], "weight_dtype": "int8_nd", "x": [64, 128]}, {"weight": [1, 192, 128], "weight_dtype": "int32_w4a8_nd", "x": [64, 192]}]`

## grouped_matmul_swiglu_quant
- status: `blocked`
- reason: Host baseline still depends on local TBE Python modules before NZ-format materialization can be probed.
- entrypoint: `torch_npu.npu_grouped_matmul_swiglu_quant`

## grouped_matmul_swiglu_quant_v2
- status: `blocked`
- reason: Host baseline requires list-valued low-precision weight tensors plus FP8/scale contracts from the upstream ACLNN example. A stable minimal Python baseline slice is not reproduced yet.
- entrypoint: `torch_npu.npu_grouped_matmul_swiglu_quant_v2`
- probe_shapes: `[{"group_list": [8], "output": [2048, 2048], "output_scale": [2048, 32, 2], "weight_list_item": [8, 7168, 4096], "weight_scale_list_item": [8, 112, 4096, 2], "x": [2048, 7168], "x_scale": [2048, 112, 2]}]`

## quant_grouped_matmul_inplace_add
- status: `blocked`
- reason: No torch_npu Python entrypoint is exposed on this host. The upstream kernel currently exists only through ACLNN/C++ host interfaces in ops-transformer tests and examples.
- entrypoint: `aclnnQuantGroupedMatmulInplaceAdd`
- probe_shapes: `[{"group_list": [4], "scale1": [4], "scale2": [4, 128], "x1": [512, 96], "x2": [512, 128], "y": [4, 96, 128]}]`
