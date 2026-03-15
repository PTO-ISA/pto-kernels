#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 32;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 0;
  int32_t v11 = 1;
  int32_t v12 = 128;
  int32_t v13 = 32;
  int32_t v14 = 4;
  int32_t v15 = 2;
  int32_t v16 = 3;
  int64_t v17 = 0;
  int64_t v18 = 2048;
  using T = float;
  size_t v19 = (size_t) v14;

  #if defined(__DAV_CUBE__)
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v17);
  Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v17);
  Tile<TileType::Right, bfloat16_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v17);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v26;
  TASSIGN(v26, v17);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v27 = (size_t) ((int32_t) (int64_t) v20); v27 < v19; v27 += (size_t) ((int32_t) (int64_t) v21)) {
    int32_t v28 = (int32_t) v27;
    int32_t v29 = (int32_t) ((uint32_t) (v28 == v16 ? v16 : v28 == v15 ? v15 : (v28 == v11 ? v11 : v10)) * (uint32_t) v13);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v30 = (size_t) v10; v30 < v19; v30 += (size_t) v11) {
      int32_t v31 = (int32_t) v30;
      int32_t v32 = (int32_t) ((uint32_t) v31 * (uint32_t) v13);
      pto::Shape<1, 1, 1, 32, 32> v33 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<4096, 4096, 4096, 128, 1> v34 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v29 * (unsigned) v12 + (unsigned) v32 * (unsigned) v11), v33, v34);
      pto::Shape<1, 1, 1, 32, 128> v36 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v37 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v9 + (unsigned) v32 * (unsigned) v12 + v9 * (unsigned) v11), v36, v37);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v22, v35);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v23, v38);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v24, v22);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v25, v23);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v31 == v10) {
        TMATMUL(v26, v24, v25);
      } else {
        TMATMUL_ACC(v26, v26, v24, v25);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v39 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v40 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v9 + (unsigned) v29 * (unsigned) v12 + v9 * (unsigned) v11), v39, v40);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v41, v26);
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

