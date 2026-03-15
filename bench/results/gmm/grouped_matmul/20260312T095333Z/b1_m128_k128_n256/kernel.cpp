#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 2048;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 0;
  int32_t v15 = 1;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 16;
  int32_t v19 = 32;
  int32_t v20 = 4;
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
  int64_t v34 = 0;
  int64_t v35 = 1024;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v36 = get_block_idx();
  int64_t v37 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v34);
  Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v35);
  Tile<TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v34);
  Tile<TileType::Right, bfloat16_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v34);
  Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 1024, PadValue::Null> v42;
  TASSIGN(v42, v34);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v43 = (size_t) ((int32_t) (int64_t) v36); v43 < ((size_t) v18); v43 += (size_t) ((int32_t) (int64_t) v37)) {
    int32_t v44 = (int32_t) v43;
    int32_t v45 = v44 == v15 ? v15 : v14;
    bool v46 = v44 == v21;
    bool v47 = v44 == v22;
    bool v48 = v44 == v20;
    bool v49 = v44 == v23;
    bool v50 = v44 == v24;
    bool v51 = v44 == v25;
    bool v52 = v44 == v26;
    bool v53 = v44 == v27;
    bool v54 = v44 == v28;
    bool v55 = v44 == v29;
    bool v56 = v44 == v30;
    bool v57 = v44 == v31;
    bool v58 = v44 == v32;
    bool v59 = v44 == v33;
    int32_t v60 = (int32_t) ((uint32_t) (v59 ? v25 : v58 ? v24 : (v57 ? v23 : v56 ? v20 : (v55 ? v22 : v54 ? v21 : (v53 ? v15 : v52 ? v14 : (v51 ? v25 : v50 ? v24 : (v49 ? v23 : v48 ? v20 : (v47 ? v22 : v46 ? v21 : v45))))))) * (uint32_t) v18);
    int32_t v61 = (int32_t) ((uint32_t) (v59 ? v14 : v58 ? v15 : (v57 ? v14 : v56 ? v15 : (v55 ? v14 : v54 ? v15 : (v53 ? v14 : v52 ? v15 : (v51 ? v15 : v50 ? v14 : (v49 ? v15 : v48 ? v14 : (v47 ? v15 : v46 ? v14 : v45))))))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v62 = (size_t) v14; v62 < ((size_t) v20); v62 += (size_t) v15) {
      int32_t v63 = (int32_t) v62;
      int32_t v64 = (int32_t) ((uint32_t) v63 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 32> v65 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<2048, 2048, 2048, 128, 1> v66 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v60 * (unsigned) v16 + (unsigned) v64 * (unsigned) v15), v65, v66);
      pto::Shape<1, 1, 1, 32, 128> v68 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<8192, 8192, 8192, 256, 1> v69 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v70 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v64 * (unsigned) v17 + (unsigned) v61 * (unsigned) v15), v68, v69);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v38, v67);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v39, v70);
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
      if (v63 == v14) {
        TMATMUL(v42, v40, v41);
      } else {
        TMATMUL_ACC(v42, v42, v40, v41);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 128> v71 = pto::Shape<1, 1, 1, 16, 128>();
    pto::Stride<4096, 4096, 4096, 256, 1> v72 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v73 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v60 * (unsigned) v17 + (unsigned) v61 * (unsigned) v15), v71, v72);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v73, v42);
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

