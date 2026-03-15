#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_finalize_routing_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, __gm__ half* v5, __gm__ int32_t* v6, __gm__ int32_t* v7) {
  unsigned v8 = 16;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 16;
  int32_t v13 = 1;
  int32_t v14 = 4;
  int64_t v15 = 0;
  int64_t v16 = 32;
  int64_t v17 = 64;
  int64_t v18 = 96;
  int64_t v19 = 128;
  int64_t v20 = 160;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v21 = get_block_idx();
  int64_t v22 = get_block_num();
  int32_t v23 = (int32_t) ((int64_t) v22);
  int32_t v24 = v12 / v23;
  int32_t v25 = v12 % v23 != v11 && v12 < v11 == v23 < v11 ? v24 + v13 : v24;
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v21) * (uint32_t) v25);
  int32_t v27 = (int32_t) ((uint32_t) v26 + (uint32_t) v25);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v15);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v16);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v17);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v18);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v19);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v20);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v34 = (size_t) v26; v34 < ((size_t) ((uint32_t) v27 < (uint32_t) v12 ? v27 : v12)); v34 += (size_t) v13) {
    int32_t v35 = (int32_t) v34;
    int32_t v36 = v6[v34];
    int32_t v37 = v7[v34];
    half v38 = v5[v34];
    pto::Shape<1, 1, 1, 1, 16> v39 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v40 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v35 * (unsigned) v12 + v10 * (unsigned) v13), v39, v40);
    pto::Shape<1, 1, 1, 1, 16> v42 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v43 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v35 * (unsigned) v12 + v10 * (unsigned) v13), v42, v43);
    pto::Shape<1, 1, 1, 1, 16> v45 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v46 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v36 * (unsigned) v12 + v10 * (unsigned) v13), v45, v46);
    pto::Shape<1, 1, 1, 1, 16> v48 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v49 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v4 + (v10 + (unsigned) v37 * (unsigned) v12 + v10 * (unsigned) v13), v48, v49);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v28, v41);
    TLOAD(v29, v47);
    TLOAD(v30, v50);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v32, v29, v30);
    TEXPANDS(v31, v38);
    pipe_barrier(PIPE_V);
    TMUL(v32, v32, v31);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v33, v28, v32);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v44, v33);
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

