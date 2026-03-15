#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void all_gather_matmul_dense(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 8;
  int32_t v13 = 4;
  int32_t v14 = 64;
  int32_t v15 = 32;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int32_t v20 = 2;
  int32_t v21 = 3;
  int32_t v22 = 5;
  int32_t v23 = 6;
  int32_t v24 = 7;
  int64_t v25 = 16384;
  int64_t v26 = 0;
  using T = float;
  int64_t v27 = get_block_idx();
  int64_t v28 = get_block_num();

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v25);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v26);
  Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v26);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v26);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v33;
  TASSIGN(v33, v26);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v34 = (size_t) ((int32_t) (int64_t) v27); v34 < ((size_t) v12); v34 += (size_t) ((int32_t) (int64_t) v28)) {
    int32_t v35 = (int32_t) v34;
    bool v36 = v35 == v13;
    bool v37 = v35 == v22;
    bool v38 = v35 == v23;
    bool v39 = v35 == v24;
    int32_t v40 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) (v39 ? v18 : v38 ? v18 : (v37 ? v18 : v36 ? v18 : v19)) * (uint32_t) v16) + (uint32_t) ((int32_t) (uint32_t) (v39 ? v21 : v38 ? v20 : (v37 ? v18 : v36 ? v19 : (v35 == v21 ? v21 : v35 == v20 ? v20 : (v35 == v18 ? v18 : v19)))) * (uint32_t) v15));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v41 = (size_t) v19; v41 < ((size_t) v13); v41 += (size_t) v18) {
      int32_t v42 = (int32_t) v41;
      int32_t v43 = (int32_t) ((uint32_t) v42 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 32, 64> v44 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v45 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v40 * (unsigned) v17 + (unsigned) v43 * (unsigned) v18), v44, v45);
      pto::Shape<1, 1, 1, 64, 128> v47 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<8192, 8192, 8192, 128, 1> v48 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v49 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v43 * (unsigned) v16 + v11 * (unsigned) v18), v47, v48);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v29, v46);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v30, v49);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v31, v29);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v32, v30);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v42 == v19) {
        TMATMUL(v33, v31, v32);
      } else {
        TMATMUL_ACC(v33, v33, v31, v32);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v50 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v51 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v52 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v40 * (unsigned) v16 + v11 * (unsigned) v18), v50, v51);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v52, v33);
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

