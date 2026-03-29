#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 2048;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 1024;
  unsigned v9 = 64;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 64;
  int32_t v16 = 128;
  int32_t v17 = 16;
  int32_t v18 = 8;
  int32_t v19 = 2;
  int32_t v20 = 3;
  int32_t v21 = 4;
  int32_t v22 = 5;
  int32_t v23 = 6;
  int32_t v24 = 7;
  int64_t v25 = 0;
  int64_t v26 = 2048;
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
  TASSIGN(v37, v25);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v38;
  __ca__ bfloat16_t* v39 = v37.data();
  uint64_t v40 = reinterpret_cast<uint64_t>(v39);
  TASSIGN(v38, v40);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v41;
  TASSIGN(v41, v25);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v42;
  __cb__ bfloat16_t* v43 = v41.data();
  uint64_t v44 = reinterpret_cast<uint64_t>(v43);
  TASSIGN(v42, v44);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v45;
  TASSIGN(v45, v25);
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
  for (int32_t v49 = (int32_t) v27; v49 < v18; v49 += (int32_t) v28) {
    bool v50 = v49 == v19;
    bool v51 = v49 == v20;
    bool v52 = v49 == v21;
    bool v53 = v49 == v22;
    bool v54 = v49 == v23;
    bool v55 = v49 == v24;
    int32_t v56 = (int32_t) ((uint32_t) (v55 ? v20 : v54 ? v20 : (v53 ? v19 : v52 ? v19 : (v51 ? v14 : v50 ? v14 : v13))) * (uint32_t) v17);
    int32_t v57 = (int32_t) ((uint32_t) (v55 ? v14 : v54 ? v13 : (v53 ? v14 : v52 ? v13 : (v51 ? v14 : v50 ? v13 : (v49 == v14 ? v14 : v13)))) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v58 = v13; v58 < v14; v58 += v14) {
      int32_t v59 = (int32_t) ((uint32_t) v58 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 64> v60 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<1024, 1024, 1024, 64, 1> v61 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v62 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v56 * (unsigned) v15 + (unsigned) v59 * (unsigned) v14), v60, v61);
      pto::Shape<1, 1, 1, 64, 64> v63 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v64 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v65 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v59 * (unsigned) v16 + (unsigned) v57 * (unsigned) v14), v63, v64);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v30, v62);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v34, v65);
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
      if (v58 == v13) {
        TMATMUL(v46, v38, v42);
      } else {
        TMATMUL_ACC(v46, v46, v38, v42);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v66 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v67 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v68 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v56 * (unsigned) v16 + (unsigned) v57 * (unsigned) v14), v66, v67);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v68, v46);
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

