#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_add_add_stage(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 128;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 128;
  int32_t v9 = 64;
  int32_t v10 = 1;
  int64_t v11 = 0;
  int64_t v12 = 512;
  int64_t v13 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) ((int64_t) v15);
  int32_t v17 = v9 / v16;
  int32_t v18 = v9 % v16 != v7 && v9 < v7 == v16 < v7 ? v17 + v10 : v17;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v14) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v11);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v12);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v13);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v24 = (size_t) v19; v24 < ((size_t) ((uint32_t) v20 < (uint32_t) v9 ? v20 : v9)); v24 += (size_t) v10) {
    int32_t v25 = (int32_t) v24;
    pto::Shape<1, 1, 1, 1, 128> v26 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v27 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v25 * (unsigned) v8 + v6 * (unsigned) v10), v26, v27);
    pto::Shape<1, 1, 1, 1, 128> v29 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v30 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v6 + (unsigned) v25 * (unsigned) v8 + v6 * (unsigned) v10), v29, v30);
    pto::Shape<1, 1, 1, 1, 128> v32 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v33 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v25 * (unsigned) v8 + v6 * (unsigned) v10), v32, v33);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v28);
    TLOAD(v22, v31);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v23, v21, v22);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v34, v23);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

