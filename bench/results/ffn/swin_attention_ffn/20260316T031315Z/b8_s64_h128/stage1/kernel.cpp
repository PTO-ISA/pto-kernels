#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 16384;
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 4;
  int32_t v9 = 128;
  int32_t v10 = 512;
  int32_t v11 = 1;
  int32_t v12 = 0;
  int32_t v13 = 2;
  int32_t v14 = 3;
  int64_t v15 = 0;
  int64_t v16 = 32768;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v15);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v20;
  __cbuf__ half* v21 = v19.data();
  uint64_t v22 = reinterpret_cast<uint64_t>(v21);
  TASSIGN(v20, v22);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v16);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v24;
  __cbuf__ half* v25 = v23.data();
  uint64_t v26 = reinterpret_cast<uint64_t>(v25);
  TASSIGN(v24, v26);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v15);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
  __ca__ half* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v15);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v32;
  __cb__ half* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v35;
  TASSIGN(v35, v15);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v36;
  __cc__ float* v37 = v35.data();
  uint64_t v38 = reinterpret_cast<uint64_t>(v37);
  TASSIGN(v36, v38);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v39 = (int32_t) v17; v39 < v8; v39 += (int32_t) v18) {
    int32_t v40 = (int32_t) ((uint32_t) (v39 == v14 ? v14 : v39 == v13 ? v13 : (v39 == v11 ? v11 : v12)) * (uint32_t) v9);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v41 = v12; v41 < v11; v41 += v11) {
      int32_t v42 = (int32_t) ((uint32_t) v41 * (uint32_t) v9);
      pto::Shape<1, 1, 1, 128, 128> v43 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v44 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v40 * (unsigned) v9 + (unsigned) v42 * (unsigned) v11), v43, v44);
      pto::Shape<1, 1, 1, 128, 128> v46 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v47 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v42 * (unsigned) v9 + v7 * (unsigned) v11), v46, v47);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v20, v45);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v24, v48);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v28, v20);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v32, v24);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v41 == v12) {
        TMATMUL(v36, v28, v32);
      } else {
        TMATMUL_ACC(v36, v36, v28, v32);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v49 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v50 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v40 * (unsigned) v9 + v7 * (unsigned) v11), v49, v50);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v51, v36);
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

