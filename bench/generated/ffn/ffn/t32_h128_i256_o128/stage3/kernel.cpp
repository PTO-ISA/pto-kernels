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
  int32_t v13 = 4;
  int32_t v14 = 64;
  int32_t v15 = 16;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 32;
  int32_t v19 = 1;
  int32_t v20 = 0;
  int32_t v21 = 2;
  int32_t v22 = 3;
  int64_t v23 = 8192;
  int64_t v24 = 0;
  using T = float;
  size_t v25 = (size_t) v13;

  #if defined(__DAV_CUBE__)
  int64_t v26 = get_block_idx();
  int64_t v27 = get_block_num();
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v23);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v24);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v24);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v24);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v32;
  TASSIGN(v32, v24);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v33 = (size_t) ((int32_t) (int64_t) v26); v33 < v25; v33 += (size_t) ((int32_t) (int64_t) v27)) {
    int32_t v34 = (int32_t) v33;
    bool v35 = v34 == v21;
    bool v36 = v34 == v22;
    int32_t v37 = (int32_t) ((uint32_t) (v36 ? v19 : v35 ? v19 : v20) * (uint32_t) v15);
    int32_t v38 = (int32_t) ((uint32_t) (v36 ? v19 : v35 ? v20 : (v34 == v19 ? v19 : v20)) * (uint32_t) v14);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v39 = (size_t) v20; v39 < v25; v39 += (size_t) v19) {
      int32_t v40 = (int32_t) v39;
      int32_t v41 = (int32_t) ((uint32_t) v40 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 16, 64> v42 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<4096, 4096, 4096, 256, 1> v43 = pto::Stride<4096, 4096, 4096, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v37 * (unsigned) v17 + (unsigned) v41 * (unsigned) v19), v42, v43);
      pto::Shape<1, 1, 1, 64, 64> v45 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v46 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v41 * (unsigned) v16 + (unsigned) v38 * (unsigned) v19), v45, v46);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v29, v47);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v31, v29);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v40 == v20) {
        TMATMUL(v32, v30, v31);
      } else {
        TMATMUL_ACC(v32, v32, v30, v31);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v48 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v49 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v37 * (unsigned) v16 + (unsigned) v38 * (unsigned) v19), v48, v49);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v50, v32);
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

