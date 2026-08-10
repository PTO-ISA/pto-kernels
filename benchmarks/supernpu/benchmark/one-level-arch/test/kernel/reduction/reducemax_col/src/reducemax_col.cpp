#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "reduction/reducemax_colvec_pto.hpp"


#ifndef DType
#define DType int32_t
#endif


#ifndef tM
#define tM 32
#endif

#ifndef tN
#define tN 64
#endif




#define gIM 256    
#define gIN 256    
// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input_buf[gIM*gIN];
//    dtype zero_buf[1*gIN];    
    dtype output_buf[1*gIN];

    dtype* input=input_buf;
//    dtype* zero=zero_buf;    
    dtype* output=output_buf;    

//    reducesum_colsum_rand<dtype, gIM, gIN, tN>(input, output);
    reducemax_col_rand<dtype, gIM, gIN, tM, tN>(input, output);

}

