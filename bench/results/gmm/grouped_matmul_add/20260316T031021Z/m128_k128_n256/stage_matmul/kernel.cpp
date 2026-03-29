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
  int32_t v22 = 4;
  int64_t v23 = 8192;
  int64_t v24 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v25 = get_block_idx();
  int64_t v26 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v23);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  __cbuf__ bfloat16_t* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v24);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  __cbuf__ bfloat16_t* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v35;
  TASSIGN(v35, v24);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v36;
  __ca__ bfloat16_t* v37 = v35.data();
  uint64_t v38 = reinterpret_cast<uint64_t>(v37);
  TASSIGN(v36, v38);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v24);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v40;
  __cb__ bfloat16_t* v41 = v39.data();
  uint64_t v42 = reinterpret_cast<uint64_t>(v41);
  TASSIGN(v40, v42);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v43;
  TASSIGN(v43, v24);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v44;
  __cc__ float* v45 = v43.data();
  uint64_t v46 = reinterpret_cast<uint64_t>(v45);
  TASSIGN(v44, v46);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v47 = (int32_t) v25; v47 < v21; v47 += (int32_t) v26) {
    int32_t v48 = (int32_t) ((uint32_t) (v47 / v22) * (uint32_t) v18);
    int32_t v49 = (int32_t) ((uint32_t) (v47 % v22) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v50 = v14; v50 < v20; v50 += v15) {
      int32_t v51 = (int32_t) ((uint32_t) v50 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 64> v52 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v53 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v48 * (unsigned) v16 + (unsigned) v51 * (unsigned) v15), v52, v53);
      pto::Shape<1, 1, 1, 64, 64> v55 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<16384, 16384, 16384, 256, 1> v56 = pto::Stride<16384, 16384, 16384, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v51 * (unsigned) v17 + (unsigned) v49 * (unsigned) v15), v55, v56);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v54);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v32, v57);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v36, v28);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v40, v32);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v50 == v14) {
        TMATMUL(v44, v36, v40);
      } else {
        TMATMUL_ACC(v44, v44, v36, v40);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v58 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v59 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v60 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v48 * (unsigned) v17 + (unsigned) v49 * (unsigned) v15), v58, v59);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v60, v44);
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

