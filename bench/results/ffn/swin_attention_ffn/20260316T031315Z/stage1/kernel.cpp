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
  int64_t v35 = 32768;
  int64_t v36 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v37 = get_block_idx();
  int64_t v38 = get_block_num();
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v39;
  TASSIGN(v39, v35);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v40;
  __cbuf__ half* v41 = v39.data();
  uint64_t v42 = reinterpret_cast<uint64_t>(v41);
  TASSIGN(v40, v42);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v43;
  TASSIGN(v43, v36);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v44;
  __cbuf__ half* v45 = v43.data();
  uint64_t v46 = reinterpret_cast<uint64_t>(v45);
  TASSIGN(v44, v46);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v47;
  TASSIGN(v47, v36);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v48;
  __ca__ half* v49 = v47.data();
  uint64_t v50 = reinterpret_cast<uint64_t>(v49);
  TASSIGN(v48, v50);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v51;
  TASSIGN(v51, v36);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v52;
  __cb__ half* v53 = v51.data();
  uint64_t v54 = reinterpret_cast<uint64_t>(v53);
  TASSIGN(v52, v54);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v55;
  TASSIGN(v55, v36);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v56;
  __cc__ float* v57 = v55.data();
  uint64_t v58 = reinterpret_cast<uint64_t>(v57);
  TASSIGN(v56, v58);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v59 = (int32_t) v37; v59 < v8; v59 += (int32_t) v38) {
    int32_t v60 = (int32_t) ((uint32_t) (v59 == v34 ? v34 : v59 == v33 ? v33 : (v59 == v32 ? v32 : v59 == v31 ? v31 : (v59 == v30 ? v30 : v59 == v29 ? v29 : (v59 == v28 ? v28 : v59 == v27 ? v27 : (v59 == v26 ? v26 : v59 == v25 ? v25 : (v59 == v24 ? v24 : v59 == v23 ? v23 : (v59 == v22 ? v22 : v59 == v21 ? v21 : (v59 == v20 ? v20 : v59 == v19 ? v19 : (v59 == v18 ? v18 : v59 == v17 ? v17 : (v59 == v16 ? v16 : v59 == v15 ? v15 : (v59 == v14 ? v14 : v59 == v13 ? v13 : (v59 == v11 ? v11 : v12)))))))))))) * (uint32_t) v9);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v61 = v12; v61 < v11; v61 += v11) {
      int32_t v62 = (int32_t) ((uint32_t) v61 * (uint32_t) v9);
      pto::Shape<1, 1, 1, 128, 128> v63 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v64 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v65 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v60 * (unsigned) v9 + (unsigned) v62 * (unsigned) v11), v63, v64);
      pto::Shape<1, 1, 1, 128, 128> v66 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v67 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v68 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v62 * (unsigned) v9 + v7 * (unsigned) v11), v66, v67);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v40, v65);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v44, v68);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v48, v40);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v52, v44);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v61 == v12) {
        TMATMUL(v56, v48, v52);
      } else {
        TMATMUL_ACC(v56, v56, v48, v52);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v69 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v70 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v71 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v60 * (unsigned) v9 + v7 * (unsigned) v11), v69, v70);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v71, v56);
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

