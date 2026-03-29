#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_add_matmul_stage(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 16384;
  unsigned v7 = 256;
  unsigned v8 = 2048;
  unsigned v9 = 128;
  unsigned v10 = 64;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 0;
  int32_t v15 = 1;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 16;
  int32_t v19 = 64;
  int32_t v20 = 2;
  int32_t v21 = 32;
  int32_t v22 = 8;
  int32_t v23 = 4;
  int32_t v24 = 3;
  int64_t v25 = 8192;
  int64_t v26 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v27 = get_block_idx();
  int64_t v28 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v25);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  __cbuf__ bfloat16_t* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v26);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v34;
  __cbuf__ bfloat16_t* v35 = v33.data();
  uint64_t v36 = reinterpret_cast<uint64_t>(v35);
  TASSIGN(v34, v36);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v26);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v38;
  __ca__ bfloat16_t* v39 = v37.data();
  uint64_t v40 = reinterpret_cast<uint64_t>(v39);
  TASSIGN(v38, v40);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v26);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v42;
  __cb__ bfloat16_t* v43 = v41.data();
  uint64_t v44 = reinterpret_cast<uint64_t>(v43);
  TASSIGN(v42, v44);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v45;
  TASSIGN(v45, v26);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v46;
  __cc__ float* v47 = v45.data();
  uint64_t v48 = reinterpret_cast<uint64_t>(v47);
  TASSIGN(v46, v48);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v49 = (int32_t) v27; v49 < v21; v49 += (int32_t) v28) {
    int32_t v50 = v49 / v22;
    int32_t v51 = v49 % v22;
    int32_t v52 = (int32_t) ((uint32_t) v50 * (uint32_t) v20);
    int32_t v53 = v50 == v24 ? (int32_t) ((uint32_t) v22 - (uint32_t) v52) : v20;
    int32_t v54 = v51 / v53;
    int32_t v55 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v52 + (uint32_t) (v51 % v53)) * (uint32_t) v18);
    int32_t v56 = (int32_t) ((uint32_t) (v50 % v20 == v15 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v23 - (uint32_t) v54) - (uint32_t) v15) : v54) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v57 = v14; v57 < v20; v57 += v15) {
      int32_t v58 = (int32_t) ((uint32_t) v57 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 64> v59 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v60 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v55 * (unsigned) v16 + (unsigned) v58 * (unsigned) v15), v59, v60);
      pto::Shape<1, 1, 1, 64, 64> v62 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<16384, 16384, 16384, 256, 1> v63 = pto::Stride<16384, 16384, 16384, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v58 * (unsigned) v17 + (unsigned) v56 * (unsigned) v15), v62, v63);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v30, v61);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v34, v64);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v38, v30);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v42, v34);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v57 == v14) {
        TMATMUL(v46, v38, v42);
      } else {
        TMATMUL_ACC(v46, v46, v38, v42);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v65 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v66 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v67 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v55 * (unsigned) v17 + (unsigned) v56 * (unsigned) v15), v65, v66);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v67, v46);
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

