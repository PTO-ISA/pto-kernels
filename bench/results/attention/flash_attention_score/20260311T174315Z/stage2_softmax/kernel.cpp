#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void flash_attention_score_stage2(__gm__ half* v1) {
  unsigned v2 = 32;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int32_t v7 = 0;
  int64_t v8 = 0;
  int64_t v9 = 64;
  int64_t v10 = 128;
  int64_t v11 = 192;
  int64_t v12 = 224;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v13;
  TASSIGN(v13, v8);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v14;
  TASSIGN(v14, v9);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v15;
  TASSIGN(v15, v10);
  Tile<TileType::Vec, half, 1, 1, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v16;
  TASSIGN(v16, v11);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v17;
  TASSIGN(v17, v12);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v18 = (size_t) v7; v18 < ((size_t) v5); v18 += (size_t) v6) {
    pto::Shape<1, 1, 1, 1, 32> v19 = pto::Shape<1, 1, 1, 1, 32>();
    pto::Stride<32, 32, 32, 32, 1> v20 = pto::Stride<32, 32, 32, 32, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v21 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) ((int32_t) v18) * (unsigned) v5 + v4 * (unsigned) v6), v19, v20);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v13, v21);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(v16, v13, v14);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v17, v16);
    pipe_barrier(PIPE_V);
    TSUB(v14, v13, v17);
    pipe_barrier(PIPE_V);
    TEXP(v14, v14);
    pipe_barrier(PIPE_V);
    TROWSUM(v16, v14, v15);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v17, v16);
    pipe_barrier(PIPE_V);
    TDIV(v15, v14, v17);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v21, v15);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

