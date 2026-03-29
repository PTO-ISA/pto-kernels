#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_init_routing_copy_indices(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  unsigned v3 = 16;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 16;
  int32_t v7 = 1;
  int64_t v8 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v9;
  TASSIGN(v9, v8);
  Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v10;
  __ubuf__ int32_t* v11 = v9.data();
  uint64_t v12 = reinterpret_cast<uint64_t>(v11);
  TASSIGN(v10, v12);
  pto::Shape<1, 1, 1, 1, 16> v13 = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> v14 = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v15 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v7), v13, v14);
  pto::Shape<1, 1, 1, 1, 16> v16 = pto::Shape<1, 1, 1, 1, 16>();
  pto::Stride<16, 16, 16, 16, 1> v17 = pto::Stride<16, 16, 16, 16, 1>();
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v18 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v7), v16, v17);
  TLOAD(v10, v15);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v18, v10);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

