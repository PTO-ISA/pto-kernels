#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_unpermute_with_routing_map_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 16;
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 0;
  int32_t v9 = 128;
  int32_t v10 = 16;
  int32_t v11 = 8;
  int32_t v12 = 1;
  int64_t v13 = 0;
  int64_t v14 = 256;
  int64_t v15 = 320;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v16 = get_block_idx();
  int64_t v17 = get_block_num();
  int32_t v18 = (int32_t) v17;
  int32_t v19 = v11 / v18;
  int32_t v20 = v11 % v18 != v8 && v11 < v8 == v18 < v8 ? v19 + v12 : v19;
  int32_t v21 = (int32_t) ((uint32_t) ((int32_t) v16) * (uint32_t) v20);
  int32_t v22 = (int32_t) ((uint32_t) v21 + (uint32_t) v20);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v24;
  __ubuf__ half* v25 = v23.data();
  uint64_t v26 = reinterpret_cast<uint64_t>(v25);
  TASSIGN(v24, v26);
  Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v14);
  Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  __ubuf__ int32_t* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v15);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  __ubuf__ half* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  pto::Shape<1, 1, 1, 1, 128> v35 = pto::Shape<1, 1, 1, 1, 128>();
  pto::Stride<128, 128, 128, 128, 1> v36 = pto::Stride<128, 128, 128, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v12), v35, v36);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v24, v37);
  for (int32_t v38 = v21; v38 < ((uint32_t) v22 < (uint32_t) v11 ? v22 : v11); v38 += v12) {
    int32_t v39 = (int32_t) ((uint32_t) v38 * (uint32_t) v10);
    pto::Shape<1, 1, 1, 1, 16> v40 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v41 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v42 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v39 * (unsigned) v12), v40, v41);
    pto::Shape<1, 1, 1, 1, 16> v43 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v44 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v39 * (unsigned) v12), v43, v44);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v28, v42);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TGATHER(v32, v24, v28);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v45, v32);
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

