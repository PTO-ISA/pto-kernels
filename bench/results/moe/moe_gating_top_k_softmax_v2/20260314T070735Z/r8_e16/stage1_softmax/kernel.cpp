#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_softmax_v2_stage(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 16;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 16;
  int32_t v8 = 8;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 32;
  int64_t v12 = 64;
  int64_t v13 = 96;
  int64_t v14 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) ((int64_t) v16);
  int32_t v18 = v8 / v17;
  int32_t v19 = v8 % v17 != v6 && v8 < v6 == v17 < v6 ? v18 + v9 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v12);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v14);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v27 = (size_t) v20; v27 < ((size_t) ((uint32_t) v21 < (uint32_t) v8 ? v21 : v8)); v27 += (size_t) v9) {
    int32_t v28 = (int32_t) v27;
    pto::Shape<1, 1, 1, 1, 16> v29 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v30 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v28 * (unsigned) v7 + v5 * (unsigned) v9), v29, v30);
    pto::Shape<1, 1, 1, 1, 16> v32 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v33 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v28 * (unsigned) v7 + v5 * (unsigned) v9), v32, v33);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v31);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TROWMAX(v25, v22, v23);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v26, v25);
    pipe_barrier(PIPE_V);
    TSUB(v23, v22, v26);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TEXP(v23, v23);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TROWSUM(v25, v23, v24);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v26, v25);
    pipe_barrier(PIPE_V);
    TDIV(v24, v23, v26);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v34, v24);
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

