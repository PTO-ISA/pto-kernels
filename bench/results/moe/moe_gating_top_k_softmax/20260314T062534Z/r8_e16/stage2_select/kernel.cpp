#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_select_stage(__gm__ half* v1, __gm__ int32_t* v2, __gm__ int32_t* v3, __gm__ half* v4, __gm__ half* v5) {
  int32_t v6 = 16;
  int32_t v7 = 0;
  int32_t v8 = 1;
  int32_t v9 = 2;
  int32_t v10 = 3;
  int32_t v11 = 4;
  int32_t v12 = 5;
  int32_t v13 = 6;
  int32_t v14 = 7;
  int32_t v15 = 8;
  int32_t v16 = 9;
  int32_t v17 = 10;
  int32_t v18 = 11;
  int32_t v19 = 12;
  int32_t v20 = 13;
  int32_t v21 = 14;
  int32_t v22 = 15;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();
  int32_t v25 = (int32_t) v24;
  int32_t v26 = v15 / v25;
  int32_t v27 = v15 % v25 != v7 && v15 < v7 == v25 < v7 ? v26 + v8 : v26;
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) v23) * (uint32_t) v27);
  int32_t v29 = (int32_t) ((uint32_t) v28 + (uint32_t) v27);
  for (int32_t v30 = v28; v30 < ((uint32_t) v29 < (uint32_t) v15 ? v29 : v15); v30 += v8) {
    int32_t v31 = (int32_t) ((uint32_t) v30 * (uint32_t) v6);
    half v32 = v5[v31];
    half v33 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v8)];
    bool v34 = v33 != v33 || v32 != v32;
    bool v35 = (v33 == v33 && v32 == v32) && v33 > v32;
    half v36 = v35 ? v33 : v32;
    half v37 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v9)];
    bool v38 = v37 != v37 || v36 != v36;
    bool v39 = (v37 == v37 && v36 == v36) && v37 > v36;
    half v40 = v39 ? v37 : v36;
    half v41 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v10)];
    bool v42 = v41 != v41 || v40 != v40;
    bool v43 = (v41 == v41 && v40 == v40) && v41 > v40;
    half v44 = v43 ? v41 : v40;
    half v45 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v11)];
    bool v46 = v45 != v45 || v44 != v44;
    bool v47 = (v45 == v45 && v44 == v44) && v45 > v44;
    half v48 = v47 ? v45 : v44;
    half v49 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v12)];
    bool v50 = v49 != v49 || v48 != v48;
    bool v51 = (v49 == v49 && v48 == v48) && v49 > v48;
    half v52 = v51 ? v49 : v48;
    half v53 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v13)];
    bool v54 = v53 != v53 || v52 != v52;
    bool v55 = (v53 == v53 && v52 == v52) && v53 > v52;
    half v56 = v55 ? v53 : v52;
    half v57 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v14)];
    bool v58 = v57 != v57 || v56 != v56;
    bool v59 = (v57 == v57 && v56 == v56) && v57 > v56;
    half v60 = v59 ? v57 : v56;
    half v61 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v15)];
    bool v62 = v61 != v61 || v60 != v60;
    bool v63 = (v61 == v61 && v60 == v60) && v61 > v60;
    half v64 = v63 ? v61 : v60;
    half v65 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v16)];
    bool v66 = v65 != v65 || v64 != v64;
    bool v67 = (v65 == v65 && v64 == v64) && v65 > v64;
    half v68 = v67 ? v65 : v64;
    half v69 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v17)];
    bool v70 = v69 != v69 || v68 != v68;
    bool v71 = (v69 == v69 && v68 == v68) && v69 > v68;
    half v72 = v71 ? v69 : v68;
    half v73 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v18)];
    bool v74 = v73 != v73 || v72 != v72;
    bool v75 = (v73 == v73 && v72 == v72) && v73 > v72;
    half v76 = v75 ? v73 : v72;
    half v77 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v19)];
    bool v78 = v77 != v77 || v76 != v76;
    bool v79 = (v77 == v77 && v76 == v76) && v77 > v76;
    half v80 = v79 ? v77 : v76;
    half v81 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v20)];
    bool v82 = v81 != v81 || v80 != v80;
    bool v83 = (v81 == v81 && v80 == v80) && v81 > v80;
    half v84 = v83 ? v81 : v80;
    half v85 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v21)];
    bool v86 = v85 != v85 || v84 != v84;
    bool v87 = (v85 == v85 && v84 == v84) && v85 > v84;
    half v88 = v87 ? v85 : v84;
    half v89 = v5[(int32_t) ((uint32_t) v31 + (uint32_t) v22)];
    bool v90 = v89 != v89 || v88 != v88;
    int32_t v91 = (v89 == v89 && v88 == v88) && v89 > v88 ? v22 : (v87 ? v21 : v83 ? v20 : (v79 ? v19 : v75 ? v18 : (v71 ? v17 : v67 ? v16 : (v63 ? v15 : v59 ? v14 : (v55 ? v13 : v51 ? v12 : (v47 ? v11 : v43 ? v10 : (v39 ? v9 : v35 ? v8 : v7)))))));
    half v92 = v4[(int32_t) ((uint32_t) v31 + (uint32_t) v91)];
    v1[v30] = v92;
    v2[v30] = v91;
    v3[v30] = v30;
  }
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

