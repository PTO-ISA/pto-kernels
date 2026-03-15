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
  int32_t v12 = 4;
  int32_t v13 = 32;
  int32_t v14 = 64;
  int32_t v15 = 256;
  int32_t v16 = 128;
  int32_t v17 = 1;
  int32_t v18 = 0;
  int32_t v19 = 2;
  int32_t v20 = 3;
  int64_t v21 = 0;
  int64_t v22 = 4096;
  using T = float;
  size_t v23 = (size_t) v12;
  int64_t v24 = get_block_idx();
  int64_t v25 = get_block_num();

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v21);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v22);
  Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v21);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v21);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v30;
  TASSIGN(v30, v21);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v31 = (size_t) ((int32_t) (int64_t) v24); v31 < v23; v31 += (size_t) ((int32_t) (int64_t) v25)) {
    int32_t v32 = (int32_t) v31;
    bool v33 = v32 == v19;
    bool v34 = v32 == v20;
    int32_t v35 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) (v34 ? v18 : v33 ? v18 : v17) * (uint32_t) v14) + (uint32_t) ((int32_t) (uint32_t) (v34 ? v17 : v33 ? v18 : (v32 == v17 ? v17 : v18)) * (uint32_t) v13));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v36 = (size_t) v18; v36 < v23; v36 += (size_t) v17) {
      int32_t v37 = (int32_t) v36;
      int32_t v38 = (int32_t) ((uint32_t) v37 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 32, 64> v39 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v40 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v35 * (unsigned) v15 + (unsigned) v38 * (unsigned) v17), v39, v40);
      pto::Shape<1, 1, 1, 64, 128> v42 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<8192, 8192, 8192, 128, 1> v43 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v38 * (unsigned) v16 + v11 * (unsigned) v17), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v26, v41);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v27, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v28, v26);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v29, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v37 == v18) {
        TMATMUL(v30, v28, v29);
      } else {
        TMATMUL_ACC(v30, v30, v28, v29);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v45 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v46 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v35 * (unsigned) v16 + v11 * (unsigned) v17), v45, v46);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v47, v30);
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

