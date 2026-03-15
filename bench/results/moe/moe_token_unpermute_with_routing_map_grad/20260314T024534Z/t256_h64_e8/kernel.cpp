#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_unpermute_with_routing_map_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ int32_t* v4) {
  unsigned v5 = 64;
  unsigned v6 = 16384;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 16384;
  int32_t v10 = 64;
  int32_t v11 = 256;
  int32_t v12 = 1;
  int32_t v13 = 0;
  half v14 = 0.0f;
  int64_t v15 = 0;
  int64_t v16 = 32768;
  int64_t v17 = 33024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v18 = get_block_idx();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int64_t v20 = get_block_num();
  int32_t v21 = (int32_t) ((int64_t) v20);
  int32_t v22 = v11 / v21;
  int32_t v23 = v11 % v21 != v13 && v11 < v13 == v21 < v13 ? v22 + v12 : v22;
  int32_t v24 = (int32_t) ((uint32_t) v19 * (uint32_t) v23);
  int32_t v25 = (int32_t) ((uint32_t) v24 + (uint32_t) v23);
  Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v17);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v19 == v13) {
    v2[v13] = v14;
  }
  pto::Shape<1, 1, 1, 1, 16384> v29 = pto::Shape<1, 1, 1, 1, 16384>();
  pto::Stride<16384, 16384, 16384, 16384, 1> v30 = pto::Stride<16384, 16384, 16384, 16384, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v3 + (v8 + v8 * (unsigned) v12), v29, v30);
  TLOAD(v26, v31);
  for (size_t v32 = (size_t) v24; v32 < ((size_t) ((uint32_t) v25 < (uint32_t) v11 ? v25 : v11)); v32 += (size_t) v12) {
    int32_t v33 = (int32_t) ((uint32_t) ((int32_t) v32) * (uint32_t) v10);
    pto::Shape<1, 1, 1, 1, 64> v34 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v35 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v36 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v4 + (v8 + (unsigned) v33 * (unsigned) v12), v34, v35);
    pto::Shape<1, 1, 1, 1, 64> v37 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v38 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v33 * (unsigned) v12), v37, v38);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v27, v36);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TGATHER(v28, v26, v27);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v39, v28);
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

