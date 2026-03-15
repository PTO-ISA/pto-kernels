#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  unsigned v2 = 32;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 0;
  int32_t v6 = 32;
  int32_t v7 = 1;
  int64_t v8 = 0;
  int64_t v9 = 64;
  int64_t v10 = 128;
  int64_t v11 = 192;
  int64_t v12 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v13 = get_block_idx();
  int64_t v14 = get_block_num();
  int32_t v15 = (int32_t) ((int64_t) v14);
  int32_t v16 = v6 / v15;
  int32_t v17 = v6 % v15 != v5 && v6 < v5 == v15 < v5 ? v16 + v7 : v16;
  int32_t v18 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v13) * (uint32_t) v17);
  int32_t v19 = (int32_t) ((uint32_t) v18 + (uint32_t) v17);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v9);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v12);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v25 = (size_t) v18; v25 < ((size_t) ((uint32_t) v19 < (uint32_t) v6 ? v19 : v6)); v25 += (size_t) v7) {
    pto::Shape<1, 1, 1, 1, 32> v26 = pto::Shape<1, 1, 1, 1, 32>();
    pto::Stride<32, 32, 32, 32, 1> v27 = pto::Stride<32, 32, 32, 32, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) ((int32_t) v25) * (unsigned) v6 + v4 * (unsigned) v7), v26, v27);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v20, v28);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(v23, v20, v21);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v24, v23);
    pipe_barrier(PIPE_V);
    TSUB(v21, v20, v24);
    pipe_barrier(PIPE_V);
    TEXP(v21, v21);
    pipe_barrier(PIPE_V);
    TROWSUM(v23, v21, v22);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v24, v23);
    pipe_barrier(PIPE_V);
    TDIV(v22, v21, v24);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v28, v22);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

