#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_relu_stage(__gm__ half* v1) {
  unsigned v2 = 256;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 0;
  int32_t v6 = 32;
  int32_t v7 = 256;
  int32_t v8 = 1;
  int64_t v9 = 0;
  int64_t v10 = 512;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v11 = get_block_idx();
  int64_t v12 = get_block_num();
  int32_t v13 = (int32_t) ((int64_t) v12);
  int32_t v14 = v6 / v13;
  int32_t v15 = v6 % v13 != v5 && v6 < v5 == v13 < v5 ? v14 + v8 : v14;
  int32_t v16 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v11) * (uint32_t) v15);
  int32_t v17 = (int32_t) ((uint32_t) v16 + (uint32_t) v15);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v9);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v10);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v20 = (size_t) v16; v20 < ((size_t) ((uint32_t) v17 < (uint32_t) v6 ? v17 : v6)); v20 += (size_t) v8) {
    pto::Shape<1, 1, 1, 1, 256> v21 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v22 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) ((int32_t) v20) * (unsigned) v7 + v4 * (unsigned) v8), v21, v22);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TRELU(v19, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v23, v19);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

