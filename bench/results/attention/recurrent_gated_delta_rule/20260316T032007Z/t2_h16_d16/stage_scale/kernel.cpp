#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void recurrent_scale_out(__gm__ bfloat16_t* v1, __gm__ float* v2) {
  RoundMode v3 = RoundMode::CAST_ROUND;
  unsigned v4 = 16;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  float v8 = 0.25f;
  int32_t v9 = 16;
  int32_t v10 = 32;
  int32_t v11 = 1;
  int64_t v12 = 0;
  int64_t v13 = 64;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) v15;
  int32_t v17 = v10 / v16;
  int32_t v18 = v10 % v16 != v7 && v10 < v7 == v16 < v7 ? v17 + v11 : v17;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) v14) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v12);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v22;
  __ubuf__ float* v23 = v21.data();
  uint64_t v24 = reinterpret_cast<uint64_t>(v23);
  TASSIGN(v22, v24);
  Tile<TileType::Vec, bfloat16_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Vec, bfloat16_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v26;
  __ubuf__ bfloat16_t* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (int32_t v29 = v19; v29 < ((uint32_t) v20 < (uint32_t) v10 ? v20 : v10); v29 += v11) {
    pto::Shape<1, 1, 1, 1, 16> v30 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v31 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v29 * (unsigned) v9 + v6 * (unsigned) v11), v30, v31);
    pto::Shape<1, 1, 1, 1, 16> v33 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v34 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v35 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v29 * (unsigned) v9 + v6 * (unsigned) v11), v33, v34);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v32);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMULS(v22, v22, v8);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v26, v22, v3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v35, v26);
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

