#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  RoundMode v2 = RoundMode::CAST_ROUND;
  unsigned v3 = 128;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 128;
  int32_t v8 = 32;
  int32_t v9 = 1;
  float v10 = 0.0883883461f;
  int64_t v11 = 0;
  int64_t v12 = 256;
  int64_t v13 = 512;
  int64_t v14 = 1024;
  int64_t v15 = 1536;
  int64_t v16 = 2048;
  int64_t v17 = 2560;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v18 = get_block_idx();
  int64_t v19 = get_block_num();
  int32_t v20 = (int32_t) ((int64_t) v19);
  int32_t v21 = v8 / v20;
  int32_t v22 = v8 % v20 != v6 && v8 < v6 == v20 < v6 ? v21 + v9 : v21;
  int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v18) * (uint32_t) v22);
  int32_t v24 = (int32_t) ((uint32_t) v23 + (uint32_t) v22);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v11);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v12);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v13);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v14);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v15);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v16);
  Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v17);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v32 = (size_t) v23; v32 < ((size_t) ((uint32_t) v24 < (uint32_t) v8 ? v24 : v8)); v32 += (size_t) v9) {
    pto::Shape<1, 1, 1, 1, 128> v33 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v34 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) ((int32_t) v32) * (unsigned) v7 + v5 * (unsigned) v9), v33, v34);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v25, v35);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v27, v25, v2);
    pipe_barrier(PIPE_V);
    TMULS(v27, v27, v10);
    pipe_barrier(PIPE_V);
    TROWMAX(v30, v27, v28);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v31, v30);
    pipe_barrier(PIPE_V);
    TSUB(v28, v27, v31);
    pipe_barrier(PIPE_V);
    TEXP(v28, v28);
    pipe_barrier(PIPE_V);
    TROWSUM(v30, v28, v29);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v31, v30);
    pipe_barrier(PIPE_V);
    TDIV(v29, v28, v31);
    pipe_barrier(PIPE_V);
    TCVT(v26, v29, v2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v35, v26);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

