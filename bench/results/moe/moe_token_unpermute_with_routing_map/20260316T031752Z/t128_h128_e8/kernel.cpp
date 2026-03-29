#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_unpermute_with_routing_map_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
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
  int32_t v17 = (int32_t) v16;
  int32_t v18 = v10 / v17;
  int32_t v19 = v10 % v17 != v8 && v10 < v8 == v17 < v8 ? v18 + v11 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v12);
  Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null> v23;
  __ubuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v13);
  Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  __ubuf__ int32_t* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v31;
  __ubuf__ half* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  pto::Shape<1, 1, 1, 1, 16384> v34 = pto::Shape<1, 1, 1, 1, 16384>();
  pto::Stride<16384, 16384, 16384, 16384, 1> v35 = pto::Stride<16384, 16384, 16384, 16384, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16384>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v11), v34, v35);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v23, v36);
  for (int32_t v37 = v20; v37 < ((uint32_t) v21 < (uint32_t) v10 ? v21 : v10); v37 += v11) {
    int32_t v38 = (int32_t) ((uint32_t) v37 * (uint32_t) v10);
    pto::Shape<1, 1, 1, 1, 128> v39 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v40 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v38 * (unsigned) v11), v39, v40);
    pto::Shape<1, 1, 1, 1, 128> v42 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v43 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v38 * (unsigned) v11), v42, v43);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v27, v41);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TGATHER(v31, v23, v27);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v44, v31);
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

