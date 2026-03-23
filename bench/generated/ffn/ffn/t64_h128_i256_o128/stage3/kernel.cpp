#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 2048;
  unsigned v5 = 8192;
  unsigned v6 = 128;
  unsigned v7 = 4096;
  unsigned v8 = 256;
  unsigned v9 = 64;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 8;
  int32_t v14 = 4;
  int32_t v15 = 16;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 64;
  int32_t v19 = 1;
  int32_t v20 = 0;
  int32_t v21 = 2;
  int32_t v22 = 3;
  int32_t v23 = 5;
  int32_t v24 = 6;
  int32_t v25 = 7;
  int64_t v26 = 8192;
  int64_t v27 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v28 = get_block_idx();
  int64_t v29 = get_block_num();
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v26);
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  __cbuf__ half* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v27);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v35;
  __cbuf__ half* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v38;
  TASSIGN(v38, v27);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v39;
  __ca__ half* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v42;
  TASSIGN(v42, v27);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v43;
  __cb__ half* v44 = v42.data();
  uint64_t v45 = reinterpret_cast<uint64_t>(v44);
  TASSIGN(v43, v45);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v46;
  TASSIGN(v46, v27);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v47;
  __cc__ float* v48 = v46.data();
  uint64_t v49 = reinterpret_cast<uint64_t>(v48);
  TASSIGN(v47, v49);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v50 = (int32_t) v28; v50 < v13; v50 += (int32_t) v29) {
    bool v51 = v50 == v21;
    bool v52 = v50 == v22;
    bool v53 = v50 == v14;
    bool v54 = v50 == v23;
    bool v55 = v50 == v24;
    bool v56 = v50 == v25;
    int32_t v57 = (int32_t) ((uint32_t) (v56 ? v22 : v55 ? v22 : (v54 ? v21 : v53 ? v21 : (v52 ? v19 : v51 ? v19 : v20))) * (uint32_t) v15);
    int32_t v58 = (int32_t) ((uint32_t) (v56 ? v19 : v55 ? v20 : (v54 ? v19 : v53 ? v20 : (v52 ? v19 : v51 ? v20 : (v50 == v19 ? v19 : v20)))) * (uint32_t) v18);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v59 = v20; v59 < v14; v59 += v19) {
      int32_t v60 = (int32_t) ((uint32_t) v59 * (uint32_t) v18);
      pto::Shape<1, 1, 1, 16, 64> v61 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<4096, 4096, 4096, 256, 1> v62 = pto::Stride<4096, 4096, 4096, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v63 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v57 * (unsigned) v17 + (unsigned) v60 * (unsigned) v19), v61, v62);
      pto::Shape<1, 1, 1, 64, 64> v64 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v65 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v66 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v60 * (unsigned) v16 + (unsigned) v58 * (unsigned) v19), v64, v65);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v31, v63);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v35, v66);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v39, v31);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v43, v35);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v59 == v20) {
        TMATMUL(v47, v39, v43);
      } else {
        TMATMUL_ACC(v47, v47, v39, v43);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v67 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v68 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v69 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v57 * (unsigned) v16 + (unsigned) v58 * (unsigned) v19), v67, v68);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v69, v47);
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

