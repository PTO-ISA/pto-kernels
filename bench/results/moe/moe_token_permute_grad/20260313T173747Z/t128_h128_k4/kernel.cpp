#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 128;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 512;
  int32_t v9 = 4;
  int32_t v10 = 128;
  int32_t v11 = 1;
  int64_t v12 = 0;
  int64_t v13 = 256;
  int64_t v14 = 512;
  using T = float;
  size_t v15 = (size_t) v11;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v16 = get_block_idx();
  int64_t v17 = get_block_num();
  int32_t v18 = (int32_t) ((int64_t) v17);
  int32_t v19 = v10 / v18;
  int32_t v20 = v10 % v18 != v7 && v10 < v7 == v18 < v7 ? v19 + v11 : v19;
  int32_t v21 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v16) * (uint32_t) v20);
  int32_t v22 = (int32_t) ((uint32_t) v21 + (uint32_t) v20);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  for (size_t v26 = (size_t) v21; v26 < ((size_t) ((uint32_t) v22 < (uint32_t) v10 ? v22 : v10)); v26 += v15) {
    int32_t v27 = (int32_t) v26;
    int32_t v28 = (int32_t) ((uint32_t) v27 * (uint32_t) v9);
    int32_t v29 = v3[v28];
    pto::Shape<1, 1, 1, 1, 128> v30 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v31 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v29 * (unsigned) v10 + v6 * (unsigned) v11), v30, v31);
    pto::Shape<1, 1, 1, 1, 128> v33 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v34 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v27 * (unsigned) v10 + v6 * (unsigned) v11), v33, v34);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v32);
    for (size_t v36 = v15; v36 < ((size_t) v9); v36 += v15) {
      int32_t v37 = v3[(int32_t) ((uint32_t) v28 + (uint32_t) ((int32_t) v36))];
      pto::Shape<1, 1, 1, 1, 128> v38 = pto::Shape<1, 1, 1, 1, 128>();
      pto::Stride<128, 128, 128, 128, 1> v39 = pto::Stride<128, 128, 128, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v40 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v37 * (unsigned) v10 + v6 * (unsigned) v11), v38, v39);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v24, v40);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      TADD(v25, v23, v24);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      pipe_barrier(PIPE_V);
      TMOV(v23, v25);
    };
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v35, v23);
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

