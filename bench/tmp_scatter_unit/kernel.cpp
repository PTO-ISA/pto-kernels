#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void k(__gm__ half* v1, __gm__ half* v2, __gm__ int16_t* v3) {
  unsigned v4 = 32;
  unsigned v5 = 16;
  unsigned v6 = 2;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 16;
  int32_t v10 = 2;
  int32_t v11 = 1;
  int64_t v12 = 64;
  int64_t v13 = 96;
  int64_t v14 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v15;
  TASSIGN(v15, v12);
  Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v16;
  TASSIGN(v16, v13);
  Tile<TileType::Vec, half, 2, 16, BLayout::RowMajor, 2, 16, SLayout::NoneBox, 512, PadValue::Null> v17;
  TASSIGN(v17, v14);
  pto::Shape<1, 1, 1, 2, 16> v18 = pto::Shape<1, 1, 1, 2, 16>();
  pto::Stride<32, 32, 32, 16, 1> v19 = pto::Stride<32, 32, 32, 16, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND> v20 = GlobalTensor<half, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND>(v1 + (v8 + v8 * (unsigned) v9 + v8 * (unsigned) v11), v18, v19);
  TLOAD(v17, v20);
  pto::Shape<1, 1, 1, 1, 16> v21 = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> v22 = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v8 + v8 * (unsigned) v9 + v8 * (unsigned) v11), v21, v22);
  TLOAD(v15, v23);
  pto::Shape<1, 1, 1, 1, 16> v24 = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> v25 = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v26 = GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>((__gm__ int16_t*) v3 + (v8 + v8 * (unsigned) v9 + v8 * (unsigned) v11), v24, v25);
  TLOAD(v16, v26);
  set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  TSCATTER(v17, v15, v16);
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  TSTORE(v20, v17);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

