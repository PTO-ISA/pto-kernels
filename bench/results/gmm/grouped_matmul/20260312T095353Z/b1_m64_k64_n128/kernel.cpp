#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 2048;
  unsigned v6 = 4096;
  unsigned v7 = 128;
  unsigned v8 = 1024;
  unsigned v9 = 64;
  unsigned v10 = 32;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 0;
  int32_t v15 = 1;
  int32_t v16 = 64;
  int32_t v17 = 128;
  int32_t v18 = 16;
  int32_t v19 = 32;
  int32_t v20 = 2;
  int32_t v21 = 4;
  int32_t v22 = 3;
  int64_t v23 = 8192;
  int64_t v24 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v25 = get_block_idx();
  int64_t v26 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v23);
  Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v24);
  Tile<TileType::Right, bfloat16_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v24);
  Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 1024, PadValue::Null> v31;
  TASSIGN(v31, v24);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v32 = (size_t) ((int32_t) (int64_t) v25); v32 < ((size_t) v21); v32 += (size_t) ((int32_t) (int64_t) v26)) {
    int32_t v33 = (int32_t) v32;
    int32_t v34 = (int32_t) ((uint32_t) (v33 == v22 ? v22 : v33 == v20 ? v20 : (v33 == v15 ? v15 : v14)) * (uint32_t) v18);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v35 = (size_t) v14; v35 < ((size_t) v20); v35 += (size_t) v15) {
      int32_t v36 = (int32_t) v35;
      int32_t v37 = (int32_t) ((uint32_t) v36 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 32> v38 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<1024, 1024, 1024, 64, 1> v39 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v40 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v34 * (unsigned) v16 + (unsigned) v37 * (unsigned) v15), v38, v39);
      pto::Shape<1, 1, 1, 32, 128> v41 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v42 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v43 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v37 * (unsigned) v17 + v13 * (unsigned) v15), v41, v42);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v27, v40);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v43);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v29, v27);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v36 == v14) {
        TMATMUL(v31, v29, v30);
      } else {
        TMATMUL_ACC(v31, v31, v29, v30);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 128> v44 = pto::Shape<1, 1, 1, 16, 128>();
    pto::Stride<2048, 2048, 2048, 128, 1> v45 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v34 * (unsigned) v17 + v13 * (unsigned) v15), v44, v45);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v46, v31);
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

