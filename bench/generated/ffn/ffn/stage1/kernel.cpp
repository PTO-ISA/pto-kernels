#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 8192;
  unsigned v6 = 256;
  unsigned v7 = 64;
  unsigned v8 = 2048;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 16;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 8;
  int32_t v15 = 4;
  int32_t v16 = 64;
  int32_t v17 = 16;
  int32_t v18 = 256;
  int32_t v19 = 128;
  int32_t v20 = 32;
  int32_t v21 = 1;
  int32_t v22 = 0;
  int32_t v23 = 2;
  int32_t v24 = 3;
  int32_t v25 = 5;
  int32_t v26 = 6;
  int32_t v27 = 7;
  int64_t v28 = 4096;
  int64_t v29 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v30 = get_block_idx();
  int64_t v31 = get_block_num();
  Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v28);
  Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v33;
  __cbuf__ half* v34 = v32.data();
  uint64_t v35 = reinterpret_cast<uint64_t>(v34);
  TASSIGN(v33, v35);
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v29);
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v37;
  __cbuf__ half* v38 = v36.data();
  uint64_t v39 = reinterpret_cast<uint64_t>(v38);
  TASSIGN(v37, v39);
  Tile<TileType::Left, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v29);
  Tile<TileType::Left, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v41;
  __ca__ half* v42 = v40.data();
  uint64_t v43 = reinterpret_cast<uint64_t>(v42);
  TASSIGN(v41, v43);
  Tile<TileType::Right, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v44;
  TASSIGN(v44, v29);
  Tile<TileType::Right, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v45;
  __cb__ half* v46 = v44.data();
  uint64_t v47 = reinterpret_cast<uint64_t>(v46);
  TASSIGN(v45, v47);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v48;
  TASSIGN(v48, v29);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v49;
  __cc__ float* v50 = v48.data();
  uint64_t v51 = reinterpret_cast<uint64_t>(v50);
  TASSIGN(v49, v51);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v52 = (int32_t) v30; v52 < v14; v52 += (int32_t) v31) {
    bool v53 = v52 == v15;
    bool v54 = v52 == v25;
    bool v55 = v52 == v26;
    bool v56 = v52 == v27;
    int32_t v57 = (int32_t) ((uint32_t) (v56 ? v21 : v55 ? v21 : (v54 ? v21 : v53 ? v21 : v22)) * (uint32_t) v17);
    int32_t v58 = (int32_t) ((uint32_t) (v56 ? v24 : v55 ? v23 : (v54 ? v21 : v53 ? v22 : (v52 == v24 ? v24 : v52 == v23 ? v23 : (v52 == v21 ? v21 : v22)))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v59 = v22; v59 < v15; v59 += v21) {
      int32_t v60 = (int32_t) ((uint32_t) v59 * (uint32_t) v20);
      pto::Shape<1, 1, 1, 16, 32> v61 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<2048, 2048, 2048, 128, 1> v62 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v63 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v13 + (unsigned) v57 * (unsigned) v19 + (unsigned) v60 * (unsigned) v21), v61, v62);
      pto::Shape<1, 1, 1, 32, 64> v64 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v65 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v66 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v13 + (unsigned) v60 * (unsigned) v18 + (unsigned) v58 * (unsigned) v21), v64, v65);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v33, v63);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v37, v66);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v41, v33);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v45, v37);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v59 == v22) {
        TMATMUL(v49, v41, v45);
      } else {
        TMATMUL_ACC(v49, v49, v41, v45);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v67 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<4096, 4096, 4096, 256, 1> v68 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v69 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v1 + (v13 + (unsigned) v57 * (unsigned) v18 + (unsigned) v58 * (unsigned) v21), v67, v68);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v69, v49);
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

