#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_add_matmul_stage(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 16384;
  unsigned v7 = 256;
  unsigned v8 = 2048;
  unsigned v9 = 128;
  unsigned v10 = 64;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 0;
  int32_t v15 = 1;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 16;
  int32_t v19 = 64;
  int32_t v20 = 2;
  int32_t v21 = 32;
  int32_t v22 = 4;
  int64_t v23 = 0;
  int64_t v24 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v25 = get_block_idx();
  int64_t v26 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v23);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v23);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v23);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v31;
  TASSIGN(v31, v23);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v32 = (size_t) ((int32_t) (int64_t) v25); v32 < ((size_t) v21); v32 += (size_t) ((int32_t) (int64_t) v26)) {
    int32_t v33 = (int32_t) v32;
    int32_t v34 = (int32_t) ((uint32_t) (v33 / v22) * (uint32_t) v18);
    int32_t v35 = (int32_t) ((uint32_t) (v33 % v22) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v36 = (size_t) v14; v36 < ((size_t) v20); v36 += (size_t) v15) {
      int32_t v37 = (int32_t) v36;
      int32_t v38 = (int32_t) ((uint32_t) v37 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 64> v39 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v40 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v34 * (unsigned) v16 + (unsigned) v38 * (unsigned) v15), v39, v40);
      pto::Shape<1, 1, 1, 64, 64> v42 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<16384, 16384, 16384, 256, 1> v43 = pto::Stride<16384, 16384, 16384, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v44 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v38 * (unsigned) v17 + (unsigned) v35 * (unsigned) v15), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v27, v41);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v44);
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
      if (v37 == v14) {
        TMATMUL(v31, v29, v30);
      } else {
        TMATMUL_ACC(v31, v31, v29, v30);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v45 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v46 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v34 * (unsigned) v17 + (unsigned) v35 * (unsigned) v15), v45, v46);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v47, v31);
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

