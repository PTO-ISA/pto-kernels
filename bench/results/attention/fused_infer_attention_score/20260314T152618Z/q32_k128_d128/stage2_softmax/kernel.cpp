#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  unsigned v2 = 128;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 0;
  int32_t v6 = 128;
  int32_t v7 = 32;
  int32_t v8 = 1;
  int64_t v9 = 0;
  int64_t v10 = 256;
  int64_t v11 = 512;
  int64_t v12 = 768;
  int64_t v13 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) ((int64_t) v15);
  int32_t v17 = v7 / v16;
  int32_t v18 = v7 % v16 != v5 && v7 < v5 == v16 < v5 ? v17 + v8 : v17;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v14) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v9);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v13);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v26 = (size_t) v19; v26 < ((size_t) ((uint32_t) v20 < (uint32_t) v7 ? v20 : v7)); v26 += (size_t) v8) {
    pto::Shape<1, 1, 1, 1, 128> v27 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v28 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) ((int32_t) v26) * (unsigned) v6 + v4 * (unsigned) v8), v27, v28);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v29);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(v24, v21, v22);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v25, v24);
    pipe_barrier(PIPE_V);
    TSUB(v22, v21, v25);
    pipe_barrier(PIPE_V);
    TEXP(v22, v22);
    pipe_barrier(PIPE_V);
    TROWSUM(v24, v22, v23);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v25, v24);
    pipe_barrier(PIPE_V);
    TDIV(v23, v22, v25);
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

