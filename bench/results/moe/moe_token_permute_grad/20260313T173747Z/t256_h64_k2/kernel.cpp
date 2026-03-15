#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 64;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 512;
  int32_t v9 = 2;
  int32_t v10 = 64;
  int32_t v11 = 256;
  int32_t v12 = 1;
  int64_t v13 = 0;
  int64_t v14 = 128;
  int64_t v15 = 256;
  using T = float;
  size_t v16 = (size_t) v12;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int32_t v20 = v11 / v19;
  int32_t v21 = v11 % v19 != v7 && v11 < v7 == v19 < v7 ? v20 + v12 : v20;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v17) * (uint32_t) v21);
  int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) v21);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v13);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  for (size_t v27 = (size_t) v22; v27 < ((size_t) ((uint32_t) v23 < (uint32_t) v11 ? v23 : v11)); v27 += v16) {
    int32_t v28 = (int32_t) v27;
    int32_t v29 = (int32_t) ((uint32_t) v28 * (uint32_t) v9);
    int32_t v30 = v3[v29];
    pto::Shape<1, 1, 1, 1, 64> v31 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v32 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v30 * (unsigned) v10 + v6 * (unsigned) v12), v31, v32);
    pto::Shape<1, 1, 1, 1, 64> v34 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v35 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v28 * (unsigned) v10 + v6 * (unsigned) v12), v34, v35);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v24, v33);
    for (size_t v37 = v16; v37 < ((size_t) v9); v37 += v16) {
      int32_t v38 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) ((int32_t) v37))];
      pto::Shape<1, 1, 1, 1, 64> v39 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v40 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v38 * (unsigned) v10 + v6 * (unsigned) v12), v39, v40);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v25, v41);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      TADD(v26, v24, v25);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      pipe_barrier(PIPE_V);
      TMOV(v24, v26);
    };
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v36, v24);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

