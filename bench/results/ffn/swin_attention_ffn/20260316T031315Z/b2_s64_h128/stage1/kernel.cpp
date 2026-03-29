#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 16384;
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 128;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 0;
  int64_t v12 = 32768;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v13 = get_block_idx();
  int64_t v14 = get_block_num();
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v15;
  TASSIGN(v15, v11);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v16;
  __cbuf__ half* v17 = v15.data();
  uint64_t v18 = reinterpret_cast<uint64_t>(v17);
  TASSIGN(v16, v18);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v12);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v20;
  __cbuf__ half* v21 = v19.data();
  uint64_t v22 = reinterpret_cast<uint64_t>(v21);
  TASSIGN(v20, v22);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v24;
  __ca__ half* v25 = v23.data();
  uint64_t v26 = reinterpret_cast<uint64_t>(v25);
  TASSIGN(v24, v26);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v11);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v28;
  __cb__ half* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v31;
  TASSIGN(v31, v11);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v32;
  __cc__ float* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v35 = (int32_t) v13; v35 < v9; v35 += (int32_t) v14) {
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v36 = v10; v36 < v9; v36 += v9) {
      int32_t v37 = (int32_t) ((uint32_t) v36 * (uint32_t) v8);
      pto::Shape<1, 1, 1, 128, 128> v38 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v39 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v40 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v8 + (unsigned) v37 * (unsigned) v9), v38, v39);
      pto::Shape<1, 1, 1, 128, 128> v41 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v42 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v43 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v37 * (unsigned) v8 + v7 * (unsigned) v9), v41, v42);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v16, v40);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v20, v43);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v24, v16);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v28, v20);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v36 == v10) {
        TMATMUL(v32, v24, v28);
      } else {
        TMATMUL_ACC(v32, v32, v24, v28);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v44 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v45 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v44, v45);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v46, v32);
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

