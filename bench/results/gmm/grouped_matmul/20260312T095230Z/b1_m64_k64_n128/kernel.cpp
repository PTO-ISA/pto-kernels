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
  int32_t v21 = 8;
  int32_t v22 = 3;
  int32_t v23 = 4;
  int32_t v24 = 5;
  int32_t v25 = 6;
  int32_t v26 = 7;
  int64_t v27 = 0;
  int64_t v28 = 1024;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v29 = get_block_idx();
  int64_t v30 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v27);
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v28);
  Tile<TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v27);
  Tile<TileType::Right, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v27);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v35;
  TASSIGN(v35, v27);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v36 = (size_t) ((int32_t) (int64_t) v29); v36 < ((size_t) v21); v36 += (size_t) ((int32_t) (int64_t) v30)) {
    int32_t v37 = (int32_t) v36;
    bool v38 = v37 == v20;
    bool v39 = v37 == v22;
    bool v40 = v37 == v23;
    bool v41 = v37 == v24;
    bool v42 = v37 == v25;
    bool v43 = v37 == v26;
    int32_t v44 = (int32_t) ((uint32_t) (v43 ? v22 : v42 ? v22 : (v41 ? v20 : v40 ? v20 : (v39 ? v15 : v38 ? v15 : v14))) * (uint32_t) v18);
    int32_t v45 = (int32_t) ((uint32_t) (v43 ? v15 : v42 ? v14 : (v41 ? v15 : v40 ? v14 : (v39 ? v15 : v38 ? v14 : (v37 == v15 ? v15 : v14)))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v46 = (size_t) v14; v46 < ((size_t) v20); v46 += (size_t) v15) {
      int32_t v47 = (int32_t) v46;
      int32_t v48 = (int32_t) ((uint32_t) v47 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 32> v49 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<1024, 1024, 1024, 64, 1> v50 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v51 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v44 * (unsigned) v16 + (unsigned) v48 * (unsigned) v15), v49, v50);
      pto::Shape<1, 1, 1, 32, 64> v52 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v53 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v48 * (unsigned) v17 + (unsigned) v45 * (unsigned) v15), v52, v53);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v31, v51);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v32, v54);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v33, v31);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v34, v32);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v47 == v14) {
        TMATMUL(v35, v33, v34);
      } else {
        TMATMUL_ACC(v35, v35, v33, v34);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v55 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v56 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v44 * (unsigned) v17 + (unsigned) v45 * (unsigned) v15), v55, v56);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v57, v35);
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

