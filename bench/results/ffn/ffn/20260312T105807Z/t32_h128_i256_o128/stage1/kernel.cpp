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
  int32_t v14 = 8;
  int32_t v15 = 4;
  int32_t v16 = 64;
  int32_t v17 = 16;
  int32_t v18 = 256;
  int32_t v19 = 128;
  int32_t v20 = 32;
  int32_t v21 = 1;
  int32_t v22 = 0;
  int32_t v23 = 2;
  int32_t v24 = 3;
  int32_t v25 = 5;
  int32_t v26 = 6;
  int32_t v27 = 7;
  int64_t v28 = 0;
  int64_t v29 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v30 = get_block_idx();
  int64_t v31 = get_block_num();
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v28);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v29);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v28);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v35;
  TASSIGN(v35, v28);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null> v36;
  TASSIGN(v36, v28);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v37 = (size_t) ((int32_t) (int64_t) v30); v37 < ((size_t) v14); v37 += (size_t) ((int32_t) (int64_t) v31)) {
    int32_t v38 = (int32_t) v37;
    bool v39 = v38 == v15;
    bool v40 = v38 == v25;
    bool v41 = v38 == v26;
    bool v42 = v38 == v27;
    int32_t v43 = (int32_t) ((uint32_t) (v42 ? v21 : v41 ? v21 : (v40 ? v21 : v39 ? v21 : v22)) * (uint32_t) v17);
    int32_t v44 = (int32_t) ((uint32_t) (v42 ? v24 : v41 ? v23 : (v40 ? v21 : v39 ? v22 : (v38 == v24 ? v24 : v38 == v23 ? v23 : (v38 == v21 ? v21 : v22)))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v45 = (size_t) v22; v45 < ((size_t) v15); v45 += (size_t) v21) {
      int32_t v46 = (int32_t) v45;
      int32_t v47 = (int32_t) ((uint32_t) v46 * (uint32_t) v20);
      pto::Shape<1, 1, 1, 16, 32> v48 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<2048, 2048, 2048, 128, 1> v49 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v43 * (unsigned) v19 + (unsigned) v47 * (unsigned) v21), v48, v49);
      pto::Shape<1, 1, 1, 32, 64> v51 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v52 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v47 * (unsigned) v18 + (unsigned) v44 * (unsigned) v21), v51, v52);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v32, v50);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v33, v53);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v34, v32);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v35, v33);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v46 == v22) {
        TMATMUL(v36, v34, v35);
      } else {
        TMATMUL_ACC(v36, v36, v34, v35);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v54 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v55 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v43 * (unsigned) v18 + (unsigned) v44 * (unsigned) v21), v54, v55);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v56, v36);
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

