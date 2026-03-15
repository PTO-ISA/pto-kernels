#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 2048;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 1024;
  unsigned v9 = 64;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 64;
  int32_t v16 = 128;
  int32_t v17 = 16;
  int32_t v18 = 8;
  int32_t v19 = 2;
  int32_t v20 = 3;
  int32_t v21 = 4;
  int32_t v22 = 5;
  int32_t v23 = 6;
  int32_t v24 = 7;
  int64_t v25 = 8192;
  int64_t v26 = 0;
  using T = float;
  size_t v27 = (size_t) v14;

  #if defined(__DAV_CUBE__)
  int64_t v28 = get_block_idx();
  int64_t v29 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v25);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v26);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v26);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v26);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v34;
  TASSIGN(v34, v26);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v35 = (size_t) ((int32_t) (int64_t) v28); v35 < ((size_t) v18); v35 += (size_t) ((int32_t) (int64_t) v29)) {
    int32_t v36 = (int32_t) v35;
    bool v37 = v36 == v19;
    bool v38 = v36 == v20;
    bool v39 = v36 == v21;
    bool v40 = v36 == v22;
    bool v41 = v36 == v23;
    bool v42 = v36 == v24;
    int32_t v43 = (int32_t) ((uint32_t) (v42 ? v20 : v41 ? v20 : (v40 ? v19 : v39 ? v19 : (v38 ? v14 : v37 ? v14 : v13))) * (uint32_t) v17);
    int32_t v44 = (int32_t) ((uint32_t) (v42 ? v14 : v41 ? v13 : (v40 ? v14 : v39 ? v13 : (v38 ? v14 : v37 ? v13 : (v36 == v14 ? v14 : v13)))) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v45 = (size_t) v13; v45 < v27; v45 += v27) {
      int32_t v46 = (int32_t) v45;
      int32_t v47 = (int32_t) ((uint32_t) v46 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 64> v48 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<1024, 1024, 1024, 64, 1> v49 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v50 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v43 * (unsigned) v15 + (unsigned) v47 * (unsigned) v14), v48, v49);
      pto::Shape<1, 1, 1, 64, 64> v51 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v52 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v47 * (unsigned) v16 + (unsigned) v44 * (unsigned) v14), v51, v52);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v30, v50);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v31, v53);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v32, v30);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v33, v31);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v46 == v13) {
        TMATMUL(v34, v32, v33);
      } else {
        TMATMUL_ACC(v34, v34, v32, v33);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v54 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v55 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v43 * (unsigned) v16 + (unsigned) v44 * (unsigned) v14), v54, v55);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v56, v34);
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

