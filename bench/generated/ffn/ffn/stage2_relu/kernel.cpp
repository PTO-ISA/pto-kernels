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
  int32_t v13 = (int32_t) v12;
  int32_t v14 = v6 / v13;
  int32_t v15 = v6 % v13 != v5 && v6 < v5 == v13 < v5 ? v14 + v8 : v14;
  int32_t v16 = (int32_t) ((uint32_t) ((int32_t) v11) * (uint32_t) v15);
  int32_t v17 = (int32_t) ((uint32_t) v16 + (uint32_t) v15);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v9);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v19;
  __ubuf__ half* v20 = v18.data();
  uint64_t v21 = reinterpret_cast<uint64_t>(v20);
  TASSIGN(v19, v21);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v23;
  __ubuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int32_t v26 = v16; v26 < ((uint32_t) v17 < (uint32_t) v6 ? v17 : v6); v26 += v8) {
    pto::Shape<1, 1, 1, 1, 256> v27 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v28 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) v26 * (unsigned) v7 + v4 * (unsigned) v8), v27, v28);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v19, v29);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TRELU(v23, v19);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v29, v23);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

