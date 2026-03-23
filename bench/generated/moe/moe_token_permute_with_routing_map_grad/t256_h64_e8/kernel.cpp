#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_with_routing_map_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ int32_t* v4) {
  unsigned v5 = 64;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 256;
  int32_t v10 = 1;
  int32_t v11 = 0;
  half v12 = 0.0f;
  int64_t v13 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int32_t v15 = (int32_t) v14;
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) v16;
  int32_t v18 = v9 / v17;
  int32_t v19 = v9 % v17 != v11 && v9 < v11 == v17 < v11 ? v18 + v10 : v18;
  int32_t v20 = (int32_t) ((uint32_t) v15 * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v13);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v23;
  __ubuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  if (v15 == v11) {
    v2[v11] = v12;
  }
  for (int32_t v26 = v20; v26 < ((uint32_t) v21 < (uint32_t) v9 ? v21 : v9); v26 += v10) {
    int32_t v27 = v4[v26];
    pto::Shape<1, 1, 1, 1, 64> v28 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v29 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v26 * (unsigned) v8 + v7 * (unsigned) v10), v28, v29);
    pto::Shape<1, 1, 1, 1, 64> v31 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v32 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v27 * (unsigned) v8 + v7 * (unsigned) v10), v31, v32);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v30);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v33, v23);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

