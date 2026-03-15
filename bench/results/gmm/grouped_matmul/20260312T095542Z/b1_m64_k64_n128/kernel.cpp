#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 2048;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 64;
  int32_t v15 = 128;
  int32_t v16 = 32;
  int32_t v17 = 2;
  int64_t v18 = 0;
  int64_t v19 = 2048;
  using T = float;
  size_t v20 = (size_t) v17;

  #if defined(__DAV_CUBE__)
  int64_t v21 = get_block_idx();
  int64_t v22 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v19);
  Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v18);
  Tile<TileType::Right, bfloat16_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v18);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v27;
  TASSIGN(v27, v18);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v28 = (size_t) ((int32_t) (int64_t) v21); v28 < v20; v28 += (size_t) ((int32_t) (int64_t) v22)) {
    int32_t v29 = (int32_t) ((uint32_t) ((int32_t) v28 == v13 ? v13 : v12) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v30 = (size_t) v12; v30 < v20; v30 += (size_t) v13) {
      int32_t v31 = (int32_t) v30;
      int32_t v32 = (int32_t) ((uint32_t) v31 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 32, 32> v33 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<2048, 2048, 2048, 64, 1> v34 = pto::Stride<2048, 2048, 2048, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v35 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v29 * (unsigned) v14 + (unsigned) v32 * (unsigned) v13), v33, v34);
      pto::Shape<1, 1, 1, 32, 128> v36 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v37 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v32 * (unsigned) v15 + v11 * (unsigned) v13), v36, v37);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v23, v35);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v24, v38);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v25, v23);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v26, v24);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v31 == v12) {
        TMATMUL(v27, v25, v26);
      } else {
        TMATMUL_ACC(v27, v27, v25, v26);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v39 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v40 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v29 * (unsigned) v15 + v11 * (unsigned) v13), v39, v40);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v41, v27);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

