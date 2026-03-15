#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_init_routing_copy_indices(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  unsigned v3 = 256;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 256;
  int32_t v7 = 1;
  int64_t v8 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, int32_t, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v9;
  TASSIGN(v9, v8);
  pto::Shape<1, 1, 1, 1, 256> v10 = pto::Shape<1, 1, 1, 1, 256>();
  pto::Stride<256, 256, 256, 256, 1> v11 = pto::Stride<256, 256, 256, 256, 1>();
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v12 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v7), v10, v11);
  pto::Shape<1, 1, 1, 1, 256> v13 = pto::Shape<1, 1, 1, 1, 256>();
  pto::Stride<256, 256, 256, 256, 1> v14 = pto::Stride<256, 256, 256, 256, 1>();
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v15 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v7), v13, v14);
  TLOAD(v9, v12);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v15, v9);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

