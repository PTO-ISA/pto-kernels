#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
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
  int64_t v24 = 0;
  int64_t v25 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v26 = get_block_idx();
  int64_t v27 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  __cbuf__ bfloat16_t* v30 = v28.data();
  uint64_t v31 = reinterpret_cast<uint64_t>(v30);
  TASSIGN(v29, v31);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v25);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
  __cbuf__ bfloat16_t* v34 = v32.data();
  uint64_t v35 = reinterpret_cast<uint64_t>(v34);
  TASSIGN(v33, v35);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v24);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v37;
  __ca__ bfloat16_t* v38 = v36.data();
  uint64_t v39 = reinterpret_cast<uint64_t>(v38);
  TASSIGN(v37, v39);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v24);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v41;
  __cb__ bfloat16_t* v42 = v40.data();
  uint64_t v43 = reinterpret_cast<uint64_t>(v42);
  TASSIGN(v41, v43);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v44;
  TASSIGN(v44, v24);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v45;
  __cc__ float* v46 = v44.data();
  uint64_t v47 = reinterpret_cast<uint64_t>(v46);
  TASSIGN(v45, v47);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v48 = (int32_t) v26; v48 < v21; v48 += (int32_t) v27) {
    int32_t v49 = v48 / v18;
    int32_t v50 = v48 % v18;
    int32_t v51 = (int32_t) ((uint32_t) v49 * (uint32_t) v20);
    int32_t v52 = v49 == v15 ? (int32_t) ((uint32_t) v23 - (uint32_t) v51) : v20;
    int32_t v53 = v50 / v52;
    int32_t v54 = (int32_t) ((uint32_t) (v49 % v20 == v15 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v22 - (uint32_t) v53) - (uint32_t) v15) : v53) * (uint32_t) v18);
    int32_t v55 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v51 + (uint32_t) (v50 % v52)) * (uint32_t) v19);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v56 = v14; v56 < v20; v56 += v15) {
      int32_t v57 = (int32_t) ((uint32_t) v56 * (uint32_t) v19);
      pto::Shape<1, 1, 1, 16, 64> v58 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v59 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v60 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v54 * (unsigned) v16 + (unsigned) v57 * (unsigned) v15), v58, v59);
      pto::Shape<1, 1, 1, 64, 64> v61 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<16384, 16384, 16384, 256, 1> v62 = pto::Stride<16384, 16384, 16384, 256, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v63 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v57 * (unsigned) v17 + (unsigned) v55 * (unsigned) v15), v61, v62);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v29, v60);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v33, v63);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v37, v29);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v41, v33);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v56 == v14) {
        TMATMUL(v45, v37, v41);
      } else {
        TMATMUL_ACC(v45, v45, v37, v41);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v64 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v65 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v66 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v54 * (unsigned) v17 + (unsigned) v55 * (unsigned) v15), v64, v65);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v66, v45);
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

