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
  int32_t v18 = 8;
  int32_t v19 = 4;
  int64_t v20 = 0;
  int64_t v21 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v22 = get_block_idx();
  int64_t v23 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v20);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
  __cbuf__ bfloat16_t* v26 = v24.data();
  uint64_t v27 = reinterpret_cast<uint64_t>(v26);
  TASSIGN(v25, v27);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v21);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  __cbuf__ bfloat16_t* v30 = v28.data();
  uint64_t v31 = reinterpret_cast<uint64_t>(v30);
  TASSIGN(v29, v31);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v20);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
  __ca__ bfloat16_t* v34 = v32.data();
  uint64_t v35 = reinterpret_cast<uint64_t>(v34);
  TASSIGN(v33, v35);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v20);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v37;
  __cb__ bfloat16_t* v38 = v36.data();
  uint64_t v39 = reinterpret_cast<uint64_t>(v38);
  TASSIGN(v37, v39);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v40;
  TASSIGN(v40, v20);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v41;
  __cc__ float* v42 = v40.data();
  uint64_t v43 = reinterpret_cast<uint64_t>(v42);
  TASSIGN(v41, v43);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v44 = (int32_t) v22; v44 < v15; v44 += (int32_t) v23) {
    int32_t v45 = v44 / v18;
    int32_t v46 = v44 % v18;
    int32_t v47 = (int32_t) ((uint32_t) v45 * (uint32_t) v19);
    int32_t v48 = v45 == v13 ? (int32_t) ((uint32_t) v18 - (uint32_t) v47) : v19;
    int32_t v49 = v46 / v48;
    int32_t v50 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v47 + (uint32_t) (v46 % v48)) * (uint32_t) v15);
    int32_t v51 = (int32_t) ((uint32_t) (v45 % v17 == v13 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v17 - (uint32_t) v49) - (uint32_t) v13) : v49) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v52 = v12; v52 < v17; v52 += v13) {
      int32_t v53 = (int32_t) ((uint32_t) v52 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 16, 64> v54 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v55 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v50 * (unsigned) v14 + (unsigned) v53 * (unsigned) v13), v54, v55);
      pto::Shape<1, 1, 1, 64, 64> v57 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v58 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v59 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v53 * (unsigned) v14 + (unsigned) v51 * (unsigned) v13), v57, v58);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v25, v56);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v29, v59);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v33, v25);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v37, v29);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v52 == v12) {
        TMATMUL(v41, v33, v37);
      } else {
        TMATMUL_ACC(v41, v41, v33, v37);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v60 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v61 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v62 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v50 * (unsigned) v14 + (unsigned) v51 * (unsigned) v13), v60, v61);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v62, v41);
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

