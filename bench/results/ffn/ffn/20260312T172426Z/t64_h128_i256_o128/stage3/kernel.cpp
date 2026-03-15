#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 2048;
  unsigned v5 = 8192;
  unsigned v6 = 128;
  unsigned v7 = 4096;
  unsigned v8 = 256;
  unsigned v9 = 64;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 8;
  int32_t v14 = 4;
  int32_t v15 = 16;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 64;
  int32_t v19 = 1;
  int32_t v20 = 0;
  int32_t v21 = 2;
  int32_t v22 = 3;
  int32_t v23 = 5;
  int32_t v24 = 6;
  int32_t v25 = 7;
  int64_t v26 = 8192;
  int64_t v27 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v28 = get_block_idx();
  int64_t v29 = get_block_num();
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v26);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v27);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v27);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v27);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v34;
  TASSIGN(v34, v27);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v35 = (size_t) ((int32_t) (int64_t) v28); v35 < ((size_t) v13); v35 += (size_t) ((int32_t) (int64_t) v29)) {
    int32_t v36 = (int32_t) v35;
    bool v37 = v36 == v21;
    bool v38 = v36 == v22;
    bool v39 = v36 == v14;
    bool v40 = v36 == v23;
    bool v41 = v36 == v24;
    bool v42 = v36 == v25;
    int32_t v43 = (int32_t) ((uint32_t) (v42 ? v22 : v41 ? v22 : (v40 ? v21 : v39 ? v21 : (v38 ? v19 : v37 ? v19 : v20))) * (uint32_t) v15);
    int32_t v44 = (int32_t) ((uint32_t) (v42 ? v19 : v41 ? v20 : (v40 ? v19 : v39 ? v20 : (v38 ? v19 : v37 ? v20 : (v36 == v19 ? v19 : v20)))) * (uint32_t) v18);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v45 = (size_t) v20; v45 < ((size_t) v14); v45 += (size_t) v19) {
      int32_t v46 = (int32_t) v45;
      int32_t v47 = (int32_t) ((uint32_t) v46 * (uint32_t) v18);
      pto::Shape<1, 1, 1, 16, 64> v48 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<4096, 4096, 4096, 256, 1> v49 = pto::Stride<4096, 4096, 4096, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v43 * (unsigned) v17 + (unsigned) v47 * (unsigned) v19), v48, v49);
      pto::Shape<1, 1, 1, 64, 64> v51 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v52 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v47 * (unsigned) v16 + (unsigned) v44 * (unsigned) v19), v51, v52);
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
      if (v46 == v20) {
        TMATMUL(v34, v32, v33);
      } else {
        TMATMUL_ACC(v34, v34, v32, v33);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v54 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v55 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v43 * (unsigned) v16 + (unsigned) v44 * (unsigned) v19), v54, v55);
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

