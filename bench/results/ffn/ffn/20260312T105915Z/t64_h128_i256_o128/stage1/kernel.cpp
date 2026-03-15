#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 8192;
  unsigned v6 = 256;
  unsigned v7 = 64;
  unsigned v8 = 2048;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 4;
  int32_t v15 = 32;
  int32_t v16 = 16;
  int32_t v17 = 256;
  int32_t v18 = 128;
  int32_t v19 = 64;
  int32_t v20 = 1;
  int32_t v21 = 0;
  int32_t v22 = 2;
  int32_t v23 = 3;
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
  int64_t v36 = 1024;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v37 = get_block_idx();
  int64_t v38 = get_block_num();
  Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v35);
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v36);
  Tile<TileType::Left, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v35);
  Tile<TileType::Right, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v42;
  TASSIGN(v42, v35);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v43;
  TASSIGN(v43, v35);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v44 = (size_t) ((int32_t) (int64_t) v37); v44 < ((size_t) v16); v44 += (size_t) ((int32_t) (int64_t) v38)) {
    int32_t v45 = (int32_t) v44;
    bool v46 = v45 == v14;
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
    int32_t v58 = (int32_t) ((uint32_t) (v57 ? v23 : v56 ? v23 : (v55 ? v23 : v54 ? v23 : (v53 ? v22 : v52 ? v22 : (v51 ? v22 : v50 ? v22 : (v49 ? v20 : v48 ? v20 : (v47 ? v20 : v46 ? v20 : v21)))))) * (uint32_t) v16);
    int32_t v59 = (int32_t) ((uint32_t) (v57 ? v23 : v56 ? v22 : (v55 ? v20 : v54 ? v21 : (v53 ? v23 : v52 ? v22 : (v51 ? v20 : v50 ? v21 : (v49 ? v23 : v48 ? v22 : (v47 ? v20 : v46 ? v21 : (v45 == v23 ? v23 : v45 == v22 ? v22 : (v45 == v20 ? v20 : v21)))))))) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v60 = (size_t) v21; v60 < ((size_t) v14); v60 += (size_t) v20) {
      int32_t v61 = (int32_t) v60;
      int32_t v62 = (int32_t) ((uint32_t) v61 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 32> v63 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<2048, 2048, 2048, 128, 1> v64 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v65 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v58 * (unsigned) v18 + (unsigned) v62 * (unsigned) v20), v63, v64);
      pto::Shape<1, 1, 1, 32, 64> v66 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v67 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v68 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v62 * (unsigned) v17 + (unsigned) v59 * (unsigned) v20), v66, v67);
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
      if (v61 == v21) {
        TMATMUL(v43, v41, v42);
      } else {
        TMATMUL_ACC(v43, v43, v41, v42);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v69 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v70 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v71 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v58 * (unsigned) v17 + (unsigned) v59 * (unsigned) v20), v69, v70);
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

