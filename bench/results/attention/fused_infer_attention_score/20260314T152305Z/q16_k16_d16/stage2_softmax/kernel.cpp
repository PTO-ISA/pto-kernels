#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  RoundMode v2 = RoundMode::CAST_ROUND;
  unsigned v3 = 16;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 16;
  int32_t v8 = 1;
  float v9 = 0.25f;
  int64_t v10 = 0;
  int64_t v11 = 32;
  int64_t v12 = 64;
  int64_t v13 = 128;
  int64_t v14 = 192;
  int64_t v15 = 256;
  int64_t v16 = 320;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int32_t v20 = v7 / v19;
  int32_t v21 = v7 % v19 != v6 && v7 < v6 == v19 < v6 ? v20 + v8 : v20;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v17) * (uint32_t) v21);
  int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) v21);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v10);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v11);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v12);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v13);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v14);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v15);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v16);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v31 = (size_t) v22; v31 < ((size_t) ((uint32_t) v23 < (uint32_t) v7 ? v23 : v7)); v31 += (size_t) v8) {
    pto::Shape<1, 1, 1, 1, 16> v32 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v33 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) ((int32_t) v31) * (unsigned) v7 + v5 * (unsigned) v8), v32, v33);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v24, v34);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v26, v24, v2);
    pipe_barrier(PIPE_V);
    TMULS(v26, v26, v9);
    pipe_barrier(PIPE_V);
    TROWMAX(v29, v26, v27);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v30, v29);
    pipe_barrier(PIPE_V);
    TSUB(v27, v26, v30);
    pipe_barrier(PIPE_V);
    TEXP(v27, v27);
    pipe_barrier(PIPE_V);
    TROWSUM(v29, v27, v28);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v30, v29);
    pipe_barrier(PIPE_V);
    TDIV(v28, v27, v30);
    pipe_barrier(PIPE_V);
    TCVT(v25, v28, v2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v34, v25);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

