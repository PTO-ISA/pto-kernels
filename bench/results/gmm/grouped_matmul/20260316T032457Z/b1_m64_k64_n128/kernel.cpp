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
  int32_t v19 = 4;
  int32_t v20 = 2;
  int64_t v21 = 8192;
  int64_t v22 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v21);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
  __cbuf__ bfloat16_t* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v22);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  __cbuf__ bfloat16_t* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v22);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v34;
  __ca__ bfloat16_t* v35 = v33.data();
  uint64_t v36 = reinterpret_cast<uint64_t>(v35);
  TASSIGN(v34, v36);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v22);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v38;
  __cb__ bfloat16_t* v39 = v37.data();
  uint64_t v40 = reinterpret_cast<uint64_t>(v39);
  TASSIGN(v38, v40);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v41;
  TASSIGN(v41, v22);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v42;
  __cc__ float* v43 = v41.data();
  uint64_t v44 = reinterpret_cast<uint64_t>(v43);
  TASSIGN(v42, v44);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v45 = (int32_t) v23; v45 < v18; v45 += (int32_t) v24) {
    int32_t v46 = v45 / v18;
    int32_t v47 = v45 % v18;
    int32_t v48 = (int32_t) ((uint32_t) v46 * (uint32_t) v19);
    int32_t v49 = v46 == v13 ? (int32_t) ((uint32_t) v19 - (uint32_t) v48) : v19;
    int32_t v50 = v47 / v49;
    int32_t v51 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v48 + (uint32_t) (v47 % v49)) * (uint32_t) v17);
    int32_t v52 = (int32_t) ((uint32_t) (v46 % v20 == v14 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v20 - (uint32_t) v50) - (uint32_t) v14) : v50) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v53 = v13; v53 < v14; v53 += v14) {
      int32_t v54 = (int32_t) ((uint32_t) v53 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 64> v55 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<1024, 1024, 1024, 64, 1> v56 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v51 * (unsigned) v15 + (unsigned) v54 * (unsigned) v14), v55, v56);
      pto::Shape<1, 1, 1, 64, 64> v58 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v59 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v60 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v54 * (unsigned) v16 + (unsigned) v52 * (unsigned) v14), v58, v59);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v26, v57);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v30, v60);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v34, v26);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v38, v30);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v53 == v13) {
        TMATMUL(v42, v34, v38);
      } else {
        TMATMUL_ACC(v42, v42, v34, v38);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v61 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v62 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v63 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v51 * (unsigned) v16 + (unsigned) v52 * (unsigned) v14), v61, v62);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v63, v42);
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

