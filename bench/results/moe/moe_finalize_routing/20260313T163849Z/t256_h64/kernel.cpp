#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_finalize_routing_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, __gm__ half* v5, __gm__ int32_t* v6, __gm__ int32_t* v7) {
  unsigned v8 = 64;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 64;
  int32_t v13 = 256;
  int32_t v14 = 1;
  int32_t v15 = 4;
  int64_t v16 = 0;
  int64_t v17 = 128;
  int64_t v18 = 256;
  int64_t v19 = 384;
  int64_t v20 = 512;
  int64_t v21 = 640;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v22 = get_block_idx();
  int64_t v23 = get_block_num();
  int32_t v24 = (int32_t) ((int64_t) v23);
  int32_t v25 = v13 / v24;
  int32_t v26 = v13 % v24 != v11 && v13 < v11 == v24 < v11 ? v25 + v14 : v25;
  int32_t v27 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) v26);
  int32_t v28 = (int32_t) ((uint32_t) v27 + (uint32_t) v26);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v16);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v17);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v18);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v19);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v20);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v21);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v35 = (size_t) v27; v35 < ((size_t) ((uint32_t) v28 < (uint32_t) v13 ? v28 : v13)); v35 += (size_t) v14) {
    int32_t v36 = (int32_t) v35;
    int32_t v37 = v6[v35];
    int32_t v38 = v7[v35];
    half v39 = v5[v35];
    pto::Shape<1, 1, 1, 1, 64> v40 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v41 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v36 * (unsigned) v12 + v10 * (unsigned) v14), v40, v41);
    pto::Shape<1, 1, 1, 1, 64> v43 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v44 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v36 * (unsigned) v12 + v10 * (unsigned) v14), v43, v44);
    pto::Shape<1, 1, 1, 1, 64> v46 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v47 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v37 * (unsigned) v12 + v10 * (unsigned) v14), v46, v47);
    pto::Shape<1, 1, 1, 1, 64> v49 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v50 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v4 + (v10 + (unsigned) v38 * (unsigned) v12 + v10 * (unsigned) v14), v49, v50);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v29, v42);
    TLOAD(v30, v48);
    TLOAD(v31, v51);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v33, v30, v31);
    TEXPANDS(v32, v39);
    pipe_barrier(PIPE_V);
    TMUL(v33, v33, v32);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v34, v29, v33);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v45, v34);
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

