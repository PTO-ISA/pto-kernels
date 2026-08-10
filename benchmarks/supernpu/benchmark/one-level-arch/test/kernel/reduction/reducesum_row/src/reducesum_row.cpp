#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
//#include "reduction/reducesum_rowvec.hpp"
#include "reduction/reducesum_rowvec_pto.hpp"



#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 128
#endif

#ifndef tNs
#define tNs 64
#endif

#ifndef gIMs
#define gIMs 1024
#endif

#ifndef gINs
#define gINs 8192
#endif    
// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input[gIMs*gINs];
    dtype output[gIMs*1];

    reducesum_trowsum_rand<dtype, gIMs, gINs, tMs, tNs>(input, output);
//每个tile只有前两个位置有数。
}
