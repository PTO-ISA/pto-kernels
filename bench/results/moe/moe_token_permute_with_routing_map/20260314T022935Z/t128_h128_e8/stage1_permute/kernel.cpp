#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_with_routing_map_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 128;
  unsigned v5 = 16384;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 0;
  int32_t v9 = 16384;
  int32_t v10 = 128;
  int32_t v11 = 1;
  int64_t v12 = 0;
  int64_t v13 = 32768;
  int64_t v14 = 33280;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) ((int64_t) v16);
  int32_t v18 = v10 / v17;
  int32_t v19 = v10 % v17 != v8 && v10 < v8 == v17 < v8 ? v18 + v11 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v12);
  Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v14);
  pto::Shape<1, 1, 1, 1, 16384> v25 = pto::Shape<1, 1, 1, 1, 16384>();
  pto::Stride<16384, 16384, 16384, 16384, 1> v26 = pto::Stride<16384, 16384, 16384, 16384, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v11), v25, v26);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v22, v27);
  for (size_t v28 = (size_t) v20; v28 < ((size_t) ((uint32_t) v21 < (uint32_t) v10 ? v21 : v10)); v28 += (size_t) v11) {
    int32_t v29 = (int32_t) ((uint32_t) ((int32_t) v28) * (uint32_t) v10);
    pto::Shape<1, 1, 1, 1, 128> v30 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v31 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v32 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v29 * (unsigned) v11), v30, v31);
    pto::Shape<1, 1, 1, 1, 128> v33 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v34 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v29 * (unsigned) v11), v33, v34);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v32);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TGATHER(v24, v22, v23);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v35, v24);
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

