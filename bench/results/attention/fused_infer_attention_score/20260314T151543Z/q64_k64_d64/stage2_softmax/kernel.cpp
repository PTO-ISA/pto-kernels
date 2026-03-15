#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  RoundMode v2 = RoundMode::CAST_ROUND;
  unsigned v3 = 64;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 64;
  int32_t v8 = 8;
  int32_t v9 = 1;
  float v10 = 0.125f;
  int64_t v11 = 0;
  int64_t v12 = 128;
  int64_t v13 = 256;
  int64_t v14 = 512;
  int64_t v15 = 768;
  int64_t v16 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int32_t v20 = v7 / v19;
  int32_t v21 = v7 % v19 != v6 && v7 < v6 == v19 < v6 ? v20 + v9 : v20;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v17) * (uint32_t) v21);
  int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) v21);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v11);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v12);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v13);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v14);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v15);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v16);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v30 = (size_t) v22; v30 < ((size_t) ((uint32_t) v23 < (uint32_t) v7 ? v23 : v7)); v30 += (size_t) v9) {
    pto::Shape<1, 1, 1, 1, 64> v31 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v32 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) ((int32_t) v30) * (unsigned) v7 + v5 * (unsigned) v9), v31, v32);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v24, v33);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v26, v24, v2);
    pipe_barrier(PIPE_V);
    TMULS(v26, v26, v10);
    pipe_barrier(PIPE_V);
    TROWMAX(v29, v26, v27);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(v27, v26, v29);
    pipe_barrier(PIPE_V);
    TEXP(v27, v27);
    pipe_barrier(PIPE_V);
    TROWSUM(v29, v27, v28);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(v28, v27, v29);
    pipe_barrier(PIPE_V);
    TCVT(v25, v28, v2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v33, v25);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

