#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_relu_stage(__gm__ half* v1) {
  unsigned v2 = 256;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 8192;
  int32_t v7 = 256;
  int32_t v8 = 1;
  int32_t v9 = 0;
  int64_t v10 = 0;
  int64_t v11 = 512;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v12;
  TASSIGN(v12, v10);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v13;
  TASSIGN(v13, v11);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v14 = (size_t) v9; v14 < ((size_t) v5); v14 += (size_t) v8) {
    pto::Shape<1, 1, 1, 1, 256> v15 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v16 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v17 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) ((int32_t) (uint32_t) ((int32_t) v14) * (uint32_t) v7) * (unsigned) v8), v15, v16);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v12, v17);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TRELU(v13, v12);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v17, v13);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

