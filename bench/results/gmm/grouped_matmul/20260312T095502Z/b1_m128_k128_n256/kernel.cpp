#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 16384;
  unsigned v7 = 256;
  unsigned v8 = 4096;
  unsigned v9 = 128;
  unsigned v10 = 64;
  unsigned v11 = 32;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 0;
  int32_t v15 = 1;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 32;
  int32_t v19 = 64;
  int32_t v20 = 2;
  int32_t v21 = 16;
  int32_t v22 = 3;
  int32_t v23 = 4;
  int32_t v24 = 5;
  int32_t v25 = 6;
  int32_t v26 = 7;
  int32_t v27 = 8;
  int32_t v28 = 9;
  int32_t v29 = 10;
  int32_t v30 = 11;
  int32_t v31 = 12;
  int32_t v32 = 13;
  int32_t v33 = 14;
  int32_t v34 = 15;
  int64_t v35 = 0;
  int64_t v36 = 4096;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v37 = get_block_idx();
  int64_t v38 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v35);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v36);
  Tile<TileType::Left, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v35);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v42;
  TASSIGN(v42, v35);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v43;
  TASSIGN(v43, v35);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v44 = (size_t) ((int32_t) (int64_t) v37); v44 < ((size_t) v21); v44 += (size_t) ((int32_t) (int64_t) v38)) {
    int32_t v45 = (int32_t) v44;
    bool v46 = v45 == v23;
    bool v47 = v45 == v24;
    bool v48 = v45 == v25;
    bool v49 = v45 == v26;
    bool v50 = v45 == v27;
    bool v51 = v45 == v28;
    bool v52 = v45 == v29;
    bool v53 = v45 == v30;
    bool v54 = v45 == v31;
    bool v55 = v45 == v32;
    bool v56 = v45 == v33;
    bool v57 = v45 == v34;
    int32_t v58 = (int32_t) ((uint32_t) (v57 ? v22 : v56 ? v22 : (v55 ? v22 : v54 ? v22 : (v53 ? v20 : v52 ? v20 : (v51 ? v20 : v50 ? v20 : (v49 ? v15 : v48 ? v15 : (v47 ? v15 : v46 ? v15 : v14)))))) * (uint32_t) v18);
    int32_t v59 = (int32_t) ((uint32_t) (v57 ? v22 : v56 ? v20 : (v55 ? v15 : v54 ? v14 : (v53 ? v22 : v52 ? v20 : (v51 ? v15 : v50 ? v14 : (v49 ? v22 : v48 ? v20 : (v47 ? v15 : v46 ? v14 : (v45 == v22 ? v22 : v45 == v20 ? v20 : (v45 == v15 ? v15 : v14)))))))) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v60 = (size_t) v14; v60 < ((size_t) v20); v60 += (size_t) v15) {
      int32_t v61 = (int32_t) v60;
      int32_t v62 = (int32_t) ((uint32_t) v61 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 32, 64> v63 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v64 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v65 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v58 * (unsigned) v16 + (unsigned) v62 * (unsigned) v15), v63, v64);
      pto::Shape<1, 1, 1, 64, 64> v66 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<16384, 16384, 16384, 256, 1> v67 = pto::Stride<16384, 16384, 16384, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v68 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v62 * (unsigned) v17 + (unsigned) v59 * (unsigned) v15), v66, v67);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v39, v65);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v40, v68);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v41, v39);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v42, v40);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v61 == v14) {
        TMATMUL(v43, v41, v42);
      } else {
        TMATMUL_ACC(v43, v43, v41, v42);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 64> v69 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<8192, 8192, 8192, 256, 1> v70 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v71 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v58 * (unsigned) v17 + (unsigned) v59 * (unsigned) v15), v69, v70);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v71, v43);
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

