#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 512;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 512;
  int32_t v8 = 1;
  int64_t v9 = 0;
  int64_t v10 = 1024;
  int64_t v11 = 3072;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null> v12;
  TASSIGN(v12, v9);
  Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null> v13;
  TASSIGN(v13, v10);
  Tile<TileType::Vec, half, 1, 512, BLayout::RowMajor, 1, 512, SLayout::NoneBox, 512, PadValue::Null> v14;
  TASSIGN(v14, v11);
  pto::Shape<1, 1, 1, 1, 512> v15 = pto::Shape<1, 1, 1, 1, 512>();
  pto::Stride<512, 512, 512, 512, 1> v16 = pto::Stride<512, 512, 512, 512, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v17 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v8), v15, v16);
  pto::Shape<1, 1, 1, 1, 512> v18 = pto::Shape<1, 1, 1, 1, 512>();
  pto::Stride<512, 512, 512, 512, 1> v19 = pto::Stride<512, 512, 512, 512, 1>();
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v20 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v3 + (v6 + v6 * (unsigned) v8), v18, v19);
  pto::Shape<1, 1, 1, 1, 512> v21 = pto::Shape<1, 1, 1, 1, 512>();
  pto::Stride<512, 512, 512, 512, 1> v22 = pto::Stride<512, 512, 512, 512, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v8), v21, v22);
  TLOAD(v12, v17);
  TLOAD(v13, v20);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TGATHER(v14, v12, v13);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v23, v14);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

