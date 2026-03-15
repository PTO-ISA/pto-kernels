#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 256;
  unsigned v7 = 64;
  unsigned v8 = 4096;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 128;
  int32_t v16 = 256;
  int32_t v17 = 32;
  int32_t v18 = 64;
  int32_t v19 = 4;
  int32_t v20 = 16;
  int32_t v21 = 2;
  int32_t v22 = 3;
  int32_t v23 = 5;
  int32_t v24 = 6;
  int32_t v25 = 7;
  int32_t v26 = 8;
  int32_t v27 = 9;
  int32_t v28 = 10;
  int32_t v29 = 11;
  int32_t v30 = 12;
  int32_t v31 = 13;
  int32_t v32 = 14;
  int32_t v33 = 15;
  int64_t v34 = 4096;
  int64_t v35 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v36 = get_block_idx();
  int64_t v37 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v34);
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v35);
  Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v35);
  Tile<TileType::Right, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v35);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v42;
  TASSIGN(v42, v35);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v43 = (size_t) ((int32_t) (int64_t) v36); v43 < ((size_t) v20); v43 += (size_t) ((int32_t) (int64_t) v37)) {
    int32_t v44 = (int32_t) v43;
    bool v45 = v44 == v19;
    bool v46 = v44 == v23;
    bool v47 = v44 == v24;
    bool v48 = v44 == v25;
    bool v49 = v44 == v26;
    bool v50 = v44 == v27;
    bool v51 = v44 == v28;
    bool v52 = v44 == v29;
    bool v53 = v44 == v30;
    bool v54 = v44 == v31;
    bool v55 = v44 == v32;
    bool v56 = v44 == v33;
    int32_t v57 = (int32_t) ((uint32_t) (v56 ? v22 : v55 ? v22 : (v54 ? v22 : v53 ? v22 : (v52 ? v21 : v51 ? v21 : (v50 ? v21 : v49 ? v21 : (v48 ? v14 : v47 ? v14 : (v46 ? v14 : v45 ? v14 : v13)))))) * (uint32_t) v17);
    int32_t v58 = (int32_t) ((uint32_t) (v56 ? v22 : v55 ? v21 : (v54 ? v14 : v53 ? v13 : (v52 ? v22 : v51 ? v21 : (v50 ? v14 : v49 ? v13 : (v48 ? v22 : v47 ? v21 : (v46 ? v14 : v45 ? v13 : (v44 == v22 ? v22 : v44 == v21 ? v21 : (v44 == v14 ? v14 : v13)))))))) * (uint32_t) v18);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v59 = (size_t) v13; v59 < ((size_t) v19); v59 += (size_t) v14) {
      int32_t v60 = (int32_t) v59;
      int32_t v61 = (int32_t) ((uint32_t) v60 * (uint32_t) v17);
      pto::Shape<1, 1, 1, 32, 32> v62 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<4096, 4096, 4096, 128, 1> v63 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v57 * (unsigned) v15 + (unsigned) v61 * (unsigned) v14), v62, v63);
      pto::Shape<1, 1, 1, 32, 64> v65 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v66 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v61 * (unsigned) v16 + (unsigned) v58 * (unsigned) v14), v65, v66);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v38, v64);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v39, v67);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v40, v38);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v41, v39);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v60 == v13) {
        TMATMUL(v42, v40, v41);
      } else {
        TMATMUL_ACC(v42, v42, v40, v41);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 64> v68 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<8192, 8192, 8192, 256, 1> v69 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v70 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v57 * (unsigned) v16 + (unsigned) v58 * (unsigned) v14), v68, v69);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v70, v42);
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

