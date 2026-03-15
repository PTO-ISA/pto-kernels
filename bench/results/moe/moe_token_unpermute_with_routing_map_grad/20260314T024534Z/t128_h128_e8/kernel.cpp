#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_unpermute_with_routing_map_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ int32_t* v4) {
  unsigned v5 = 128;
  unsigned v6 = 16384;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 16384;
  int32_t v10 = 128;
  int32_t v11 = 1;
  int32_t v12 = 0;
  half v13 = 0.0f;
  int64_t v14 = 0;
  int64_t v15 = 32768;
  int64_t v16 = 33280;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v17 = get_block_idx();
  int32_t v18 = (int32_t) ((int64_t) v17);
  int64_t v19 = get_block_num();
  int32_t v20 = (int32_t) ((int64_t) v19);
  int32_t v21 = v10 / v20;
  int32_t v22 = v10 % v20 != v12 && v10 < v12 == v20 < v12 ? v21 + v11 : v21;
  int32_t v23 = (int32_t) ((uint32_t) v18 * (uint32_t) v22);
  int32_t v24 = (int32_t) ((uint32_t) v23 + (uint32_t) v22);
  Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v18 == v12) {
    v2[v12] = v13;
  }
  pto::Shape<1, 1, 1, 1, 16384> v28 = pto::Shape<1, 1, 1, 1, 16384>();
  pto::Stride<16384, 16384, 16384, 16384, 1> v29 = pto::Stride<16384, 16384, 16384, 16384, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v3 + (v8 + v8 * (unsigned) v11), v28, v29);
  TLOAD(v25, v30);
  for (size_t v31 = (size_t) v23; v31 < ((size_t) ((uint32_t) v24 < (uint32_t) v10 ? v24 : v10)); v31 += (size_t) v11) {
    int32_t v32 = (int32_t) ((uint32_t) ((int32_t) v31) * (uint32_t) v10);
    pto::Shape<1, 1, 1, 1, 128> v33 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v34 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v8 + (unsigned) v32 * (unsigned) v11), v33, v34);
    pto::Shape<1, 1, 1, 1, 128> v36 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v37 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v32 * (unsigned) v11), v36, v37);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v26, v35);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TGATHER(v27, v25, v26);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v38, v27);
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

