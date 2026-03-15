#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 2048;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 16;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 128;
  int32_t v15 = 16;
  int32_t v16 = 64;
  int32_t v17 = 2;
  int32_t v18 = 8;
  int32_t v19 = 3;
  int32_t v20 = 4;
  int32_t v21 = 5;
  int32_t v22 = 6;
  int32_t v23 = 7;
  int64_t v24 = 0;
  int64_t v25 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v26 = get_block_idx();
  int64_t v27 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v25);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v24);
  Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v24);
  Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 1024, PadValue::Null> v32;
  TASSIGN(v32, v24);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v33 = (size_t) ((int32_t) (int64_t) v26); v33 < ((size_t) v18); v33 += (size_t) ((int32_t) (int64_t) v27)) {
    int32_t v34 = (int32_t) v33;
    int32_t v35 = (int32_t) ((uint32_t) (v34 == v23 ? v23 : v34 == v22 ? v22 : (v34 == v21 ? v21 : v34 == v20 ? v20 : (v34 == v19 ? v19 : v34 == v17 ? v17 : (v34 == v13 ? v13 : v12)))) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v36 = (size_t) v12; v36 < ((size_t) v17); v36 += (size_t) v13) {
      int32_t v37 = (int32_t) v36;
      int32_t v38 = (int32_t) ((uint32_t) v37 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 16, 64> v39 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v40 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v35 * (unsigned) v14 + (unsigned) v38 * (unsigned) v13), v39, v40);
      pto::Shape<1, 1, 1, 64, 128> v42 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<8192, 8192, 8192, 128, 1> v43 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v38 * (unsigned) v14 + v11 * (unsigned) v13), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v41);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v29, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v31, v29);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v37 == v12) {
        TMATMUL(v32, v30, v31);
      } else {
        TMATMUL_ACC(v32, v32, v30, v31);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 128> v45 = pto::Shape<1, 1, 1, 16, 128>();
    pto::Stride<2048, 2048, 2048, 128, 1> v46 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v35 * (unsigned) v14 + v11 * (unsigned) v13), v45, v46);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v47, v32);
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

