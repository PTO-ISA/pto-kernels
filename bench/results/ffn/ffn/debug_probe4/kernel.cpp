#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void ffn_dense_relu_fp16(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, __gm__ half* v5) {
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 4096;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 8;
  int32_t v14 = 4;
  int32_t v15 = 256;
  int32_t v16 = 128;
  int32_t v17 = 32;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int32_t v20 = 8192;
  int64_t v21 = 0;
  int64_t v22 = 2048;
  int64_t v23 = 512;
  int64_t v24 = 18432;
  int64_t v25 = 20480;
  int64_t v26 = 16384;
  int64_t v27 = 32768;
  using T = float;
  size_t v28 = (size_t) v19;
  size_t v29 = (size_t) v18;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v21);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v22);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v21);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v21);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null> v34;
  TASSIGN(v34, v21);
  for (size_t v35 = v28; v35 < ((size_t) v14); v35 += v29) {
    int32_t v36 = (int32_t) v35;
    int32_t v37 = (int32_t) ((uint32_t) v36 * (uint32_t) v17);
    pto::Shape<1, 1, 1, 32, 32> v38 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<4096, 4096, 4096, 128, 1> v39 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v40 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + v12 * (unsigned) v16 + (unsigned) v37 * (unsigned) v18), v38, v39);
    pto::Shape<1, 1, 1, 32, 256> v41 = pto::Shape<1, 1, 1, 32, 256>();
    pto::Stride<8192, 8192, 8192, 256, 1> v42 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v43 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v4 + (v12 + (unsigned) v37 * (unsigned) v15 + v12 * (unsigned) v18), v41, v42);
    TLOAD(v30, v40);
    TLOAD(v31, v43);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v32, v30);
    TMOV(v33, v31);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v36 == v19) {
      TMATMUL(v34, v32, v33);
    } else {
      TMATMUL_ACC(v34, v34, v32, v33);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 256> v44 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v45 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v18), v44, v45);
  TSTORE(v46, v34);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v47;
  TASSIGN(v47, v21);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v48;
  TASSIGN(v48, v23);
  for (size_t v49 = v28; v49 < ((size_t) v17); v49 += v29) {
    pto::Shape<1, 1, 1, 1, 256> v50 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v51 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v52 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) ((int32_t) (uint32_t) ((int32_t) v49) * (uint32_t) v15) * (unsigned) v18), v50, v51);
    TLOAD(v47, v52);
    TRELU(v48, v47);
    TSTORE(v52, v48);
  }
  #endif // __DAV_VEC__


  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v53;
  TASSIGN(v53, v24);
  Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v54;
  TASSIGN(v54, v25);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v55;
  TASSIGN(v55, v22);
  Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v56;
  TASSIGN(v56, v26);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v57;
  TASSIGN(v57, v27);
  for (size_t v58 = v28; v58 < ((size_t) v13); v58 += v29) {
    int32_t v59 = (int32_t) v58;
    int32_t v60 = (int32_t) ((uint32_t) v59 * (uint32_t) v17);
    pto::Shape<1, 1, 1, 32, 32> v61 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<8192, 8192, 8192, 256, 1> v62 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v63 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v15 + (unsigned) v60 * (unsigned) v18), v61, v62);
    pto::Shape<1, 1, 1, 32, 128> v64 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v65 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v66 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v5 + (v12 + (unsigned) v60 * (unsigned) v16 + v12 * (unsigned) v18), v64, v65);
    TLOAD(v53, v63);
    TLOAD(v54, v66);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v55, v53);
    TMOV(v56, v54);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    if (v59 == v19) {
      TMATMUL(v57, v55, v56);
    } else {
      TMATMUL_ACC(v57, v57, v55, v56);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pto::Shape<1, 1, 1, 32, 128> v67 = pto::Shape<1, 1, 1, 32, 128>();
  pto::Stride<4096, 4096, 4096, 128, 1> v68 = pto::Stride<4096, 4096, 4096, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v69 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v12 + v12 * (unsigned) v16 + v12 * (unsigned) v18), v67, v68);
  TSTORE(v69, v57);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  #endif // __DAV_CUBE__

  return;
}

