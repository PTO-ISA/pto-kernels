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
  int32_t v18 = 3;
  int32_t v19 = 4;
  int32_t v20 = 5;
  int32_t v21 = 6;
  int32_t v22 = 7;
  int32_t v23 = 8;
  int32_t v24 = 9;
  int32_t v25 = 10;
  int32_t v26 = 11;
  int32_t v27 = 12;
  int32_t v28 = 13;
  int32_t v29 = 14;
  int32_t v30 = 15;
  int64_t v31 = 8192;
  int64_t v32 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v33 = get_block_idx();
  int64_t v34 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v35;
  TASSIGN(v35, v31);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v32);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v32);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v32);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v39;
  TASSIGN(v39, v32);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v40 = (size_t) ((int32_t) (int64_t) v33); v40 < ((size_t) v15); v40 += (size_t) ((int32_t) (int64_t) v34)) {
    int32_t v41 = (int32_t) v40;
    int32_t v42 = v41 == v13 ? v13 : v12;
    bool v43 = v41 == v17;
    bool v44 = v41 == v18;
    bool v45 = v41 == v19;
    bool v46 = v41 == v20;
    bool v47 = v41 == v21;
    bool v48 = v41 == v22;
    bool v49 = v41 == v23;
    bool v50 = v41 == v24;
    bool v51 = v41 == v25;
    bool v52 = v41 == v26;
    bool v53 = v41 == v27;
    bool v54 = v41 == v28;
    bool v55 = v41 == v29;
    bool v56 = v41 == v30;
    int32_t v57 = (int32_t) ((uint32_t) (v56 ? v22 : v55 ? v21 : (v54 ? v20 : v53 ? v19 : (v52 ? v18 : v51 ? v17 : (v50 ? v13 : v49 ? v12 : (v48 ? v22 : v47 ? v21 : (v46 ? v20 : v45 ? v19 : (v44 ? v18 : v43 ? v17 : v42))))))) * (uint32_t) v15);
    int32_t v58 = (int32_t) ((uint32_t) (v56 ? v12 : v55 ? v13 : (v54 ? v12 : v53 ? v13 : (v52 ? v12 : v51 ? v13 : (v50 ? v12 : v49 ? v13 : (v48 ? v13 : v47 ? v12 : (v46 ? v13 : v45 ? v12 : (v44 ? v13 : v43 ? v12 : v42))))))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v59 = (size_t) v12; v59 < ((size_t) v17); v59 += (size_t) v13) {
      int32_t v60 = (int32_t) v59;
      int32_t v61 = (int32_t) ((uint32_t) v60 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 16, 64> v62 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v63 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v57 * (unsigned) v14 + (unsigned) v61 * (unsigned) v13), v62, v63);
      pto::Shape<1, 1, 1, 64, 64> v65 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v66 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v61 * (unsigned) v14 + (unsigned) v58 * (unsigned) v13), v65, v66);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v35, v64);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v36, v67);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v37, v35);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v38, v36);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v60 == v12) {
        TMATMUL(v39, v37, v38);
      } else {
        TMATMUL_ACC(v39, v39, v37, v38);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v68 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v69 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v70 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v57 * (unsigned) v14 + (unsigned) v58 * (unsigned) v13), v68, v69);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v70, v39);
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

