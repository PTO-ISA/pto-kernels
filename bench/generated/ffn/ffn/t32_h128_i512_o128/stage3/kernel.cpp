#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 2048;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 512;
  unsigned v8 = 64;
  unsigned v9 = 16;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 4;
  int32_t v13 = 8;
  int32_t v14 = 64;
  int32_t v15 = 16;
  int32_t v16 = 128;
  int32_t v17 = 512;
  int32_t v18 = 32;
  int32_t v19 = 1;
  int32_t v20 = 0;
  int32_t v21 = 2;
  int32_t v22 = 3;
  int64_t v23 = 0;
  int64_t v24 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v25 = get_block_idx();
  int64_t v26 = get_block_num();
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v23);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v23);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v23);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v31;
  TASSIGN(v31, v23);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v32 = (size_t) ((int32_t) (int64_t) v25); v32 < ((size_t) v12); v32 += (size_t) ((int32_t) (int64_t) v26)) {
    int32_t v33 = (int32_t) v32;
    bool v34 = v33 == v21;
    bool v35 = v33 == v22;
    int32_t v36 = (int32_t) ((uint32_t) (v35 ? v19 : v34 ? v19 : v20) * (uint32_t) v15);
    int32_t v37 = (int32_t) ((uint32_t) (v35 ? v19 : v34 ? v20 : (v33 == v19 ? v19 : v20)) * (uint32_t) v14);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v38 = (size_t) v20; v38 < ((size_t) v13); v38 += (size_t) v19) {
      int32_t v39 = (int32_t) v38;
      int32_t v40 = (int32_t) ((uint32_t) v39 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 16, 64> v41 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<8192, 8192, 8192, 512, 1> v42 = pto::Stride<8192, 8192, 8192, 512, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v43 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v36 * (unsigned) v17 + (unsigned) v40 * (unsigned) v19), v41, v42);
      pto::Shape<1, 1, 1, 64, 64> v44 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v45 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v40 * (unsigned) v16 + (unsigned) v37 * (unsigned) v19), v44, v45);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v27, v43);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v46);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v29, v27);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v39 == v20) {
        TMATMUL(v31, v29, v30);
      } else {
        TMATMUL_ACC(v31, v31, v29, v30);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v47 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v48 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v49 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v36 * (unsigned) v16 + (unsigned) v37 * (unsigned) v19), v47, v48);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v49, v31);
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

