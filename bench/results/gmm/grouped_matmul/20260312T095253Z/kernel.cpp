#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 64;
  unsigned v7 = 2048;
  unsigned v8 = 128;
  unsigned v9 = 32;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 128;
  int32_t v16 = 16;
  int32_t v17 = 64;
  int32_t v18 = 32;
  int32_t v19 = 4;
  int32_t v20 = 2;
  int32_t v21 = 3;
  int32_t v22 = 5;
  int32_t v23 = 6;
  int32_t v24 = 7;
  int32_t v25 = 8;
  int32_t v26 = 9;
  int32_t v27 = 10;
  int32_t v28 = 11;
  int32_t v29 = 12;
  int32_t v30 = 13;
  int32_t v31 = 14;
  int32_t v32 = 15;
  int64_t v33 = 4096;
  int64_t v34 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v35 = get_block_idx();
  int64_t v36 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v33);
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v34);
  Tile<TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v34);
  Tile<TileType::Right, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v34);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v41;
  TASSIGN(v41, v34);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v42 = (size_t) ((int32_t) (int64_t) v35); v42 < ((size_t) v16); v42 += (size_t) ((int32_t) (int64_t) v36)) {
    int32_t v43 = (int32_t) v42;
    int32_t v44 = v43 == v14 ? v14 : v13;
    bool v45 = v43 == v20;
    bool v46 = v43 == v21;
    bool v47 = v43 == v19;
    bool v48 = v43 == v22;
    bool v49 = v43 == v23;
    bool v50 = v43 == v24;
    bool v51 = v43 == v25;
    bool v52 = v43 == v26;
    bool v53 = v43 == v27;
    bool v54 = v43 == v28;
    bool v55 = v43 == v29;
    bool v56 = v43 == v30;
    bool v57 = v43 == v31;
    bool v58 = v43 == v32;
    int32_t v59 = (int32_t) ((uint32_t) (v58 ? v24 : v57 ? v23 : (v56 ? v22 : v55 ? v19 : (v54 ? v21 : v53 ? v20 : (v52 ? v14 : v51 ? v13 : (v50 ? v24 : v49 ? v23 : (v48 ? v22 : v47 ? v19 : (v46 ? v21 : v45 ? v20 : v44))))))) * (uint32_t) v16);
    int32_t v60 = (int32_t) ((uint32_t) (v58 ? v13 : v57 ? v14 : (v56 ? v13 : v55 ? v14 : (v54 ? v13 : v53 ? v14 : (v52 ? v13 : v51 ? v14 : (v50 ? v14 : v49 ? v13 : (v48 ? v14 : v47 ? v13 : (v46 ? v14 : v45 ? v13 : v44))))))) * (uint32_t) v17);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v61 = (size_t) v13; v61 < ((size_t) v19); v61 += (size_t) v14) {
      int32_t v62 = (int32_t) v61;
      int32_t v63 = (int32_t) ((uint32_t) v62 * (uint32_t) v18);
      pto::Shape<1, 1, 1, 16, 32> v64 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<2048, 2048, 2048, 128, 1> v65 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v66 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v59 * (unsigned) v15 + (unsigned) v63 * (unsigned) v14), v64, v65);
      pto::Shape<1, 1, 1, 32, 64> v67 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v68 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v69 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v63 * (unsigned) v15 + (unsigned) v60 * (unsigned) v14), v67, v68);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v37, v66);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v38, v69);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v39, v37);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v40, v38);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v62 == v13) {
        TMATMUL(v41, v39, v40);
      } else {
        TMATMUL_ACC(v41, v41, v39, v40);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v70 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v71 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v72 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v59 * (unsigned) v15 + (unsigned) v60 * (unsigned) v14), v70, v71);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v72, v41);
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

