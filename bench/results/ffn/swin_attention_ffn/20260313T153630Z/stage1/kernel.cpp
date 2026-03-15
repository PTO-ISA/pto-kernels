#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 16384;
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 24;
  int32_t v9 = 128;
  int32_t v10 = 3072;
  int32_t v11 = 1;
  int32_t v12 = 0;
  int32_t v13 = 2;
  int32_t v14 = 3;
  int32_t v15 = 4;
  int32_t v16 = 5;
  int32_t v17 = 6;
  int32_t v18 = 7;
  int32_t v19 = 8;
  int32_t v20 = 9;
  int32_t v21 = 10;
  int32_t v22 = 11;
  int32_t v23 = 12;
  int32_t v24 = 13;
  int32_t v25 = 14;
  int32_t v26 = 15;
  int32_t v27 = 16;
  int32_t v28 = 17;
  int32_t v29 = 18;
  int32_t v30 = 19;
  int32_t v31 = 20;
  int32_t v32 = 21;
  int32_t v33 = 22;
  int32_t v34 = 23;
  int64_t v35 = 0;
  int64_t v36 = 32768;
  using T = float;
  size_t v37 = (size_t) v11;

  #if defined(__DAV_CUBE__)
  int64_t v38 = get_block_idx();
  int64_t v39 = get_block_num();
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v35);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v36);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v42;
  TASSIGN(v42, v35);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v43;
  TASSIGN(v43, v35);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v44;
  TASSIGN(v44, v35);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v45 = (size_t) ((int32_t) (int64_t) v38); v45 < ((size_t) v8); v45 += (size_t) ((int32_t) (int64_t) v39)) {
    int32_t v46 = (int32_t) v45;
    int32_t v47 = (int32_t) ((uint32_t) (v46 == v34 ? v34 : v46 == v33 ? v33 : (v46 == v32 ? v32 : v46 == v31 ? v31 : (v46 == v30 ? v30 : v46 == v29 ? v29 : (v46 == v28 ? v28 : v46 == v27 ? v27 : (v46 == v26 ? v26 : v46 == v25 ? v25 : (v46 == v24 ? v24 : v46 == v23 ? v23 : (v46 == v22 ? v22 : v46 == v21 ? v21 : (v46 == v20 ? v20 : v46 == v19 ? v19 : (v46 == v18 ? v18 : v46 == v17 ? v17 : (v46 == v16 ? v16 : v46 == v15 ? v15 : (v46 == v14 ? v14 : v46 == v13 ? v13 : (v46 == v11 ? v11 : v12)))))))))))) * (uint32_t) v9);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v48 = (size_t) v12; v48 < v37; v48 += v37) {
      int32_t v49 = (int32_t) v48;
      int32_t v50 = (int32_t) ((uint32_t) v49 * (uint32_t) v9);
      pto::Shape<1, 1, 1, 128, 128> v51 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v52 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v47 * (unsigned) v9 + (unsigned) v50 * (unsigned) v11), v51, v52);
      pto::Shape<1, 1, 1, 128, 128> v54 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v55 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v50 * (unsigned) v9 + v7 * (unsigned) v11), v54, v55);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v40, v53);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v41, v56);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v42, v40);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v43, v41);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v49 == v12) {
        TMATMUL(v44, v42, v43);
      } else {
        TMATMUL_ACC(v44, v44, v42, v43);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v57 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v58 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v59 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v47 * (unsigned) v9 + v7 * (unsigned) v11), v57, v58);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v59, v44);
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

