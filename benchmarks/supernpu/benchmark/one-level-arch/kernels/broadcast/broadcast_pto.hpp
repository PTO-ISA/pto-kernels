// PTO/LinxISA v0.58 broadcast workload using the canonical TileOP API.

#include <cstdint>
#include <cstdio>

// ============================================================================
// 维度规则：从后面对齐，前面自动补 1，维度=1 可广播
// ============================================================================

// ----------------------------------------------------------------------------
// gen_offset_pto: 用 PTO ISA Tile 指令计算广播偏移 (一层编程，无 __vec__ 块)
//
// 算法 (stride 累加法，从最内维到最外维):
//   1. TCI      生成索引序列  base, base+1, ..., base+N-1
//   2. TEXPANDS 将输出偏移 tile 初始化为 0
//   3. 对每个输出维 d (从 OUT_DIM-1 到 0):
//        TREMS  coord = idx % out_shape[d]          (标量取余)
//        TDIVS  idx   = idx / out_shape[d]          (标量整除)
//        若 d 对应输入维 i = d-(OUT_DIM-IN_DIM) >= 0:
//          非广播维 (in_shape[i]!=1):
//            TMULS  tmp = coord * stride
//            TADD   out += tmp
//          stride *= in_shape[i]
//
// 注意: PTO ISA MGATHER<Coalesce::Elem> 使用元素索引 (非字节偏移),
//       所以这里不再乘 sizeof(dtype)。
// ----------------------------------------------------------------------------
template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void gen_offset_pto(
    tile_shapeOffset &out,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base,
    const size_t total_elements)
{
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");

    using off_t = typename tile_shapeOffset::DType;   // uint32_t

    tile_shapeOffset idxTile;     // 当前正在分解的线性索引
    tile_shapeOffset coordTile;   // 当前输出维坐标
    tile_shapeOffset tmpTile;     // TREMS 需要 (A2A3 要求 tmp >= 1 行, 列数 >= dst)

    // ---- Step 1: TCI 生成索引序列 [base, base+1, ..., base+N-1] ----
    TCI(idxTile, (off_t)base);

    // ---- Step 2: TEXPANDS 初始化 out = 0 ----
    TEXPANDS(out, (off_t)0);

    size_t stride = 1;

    // ---- Step 3: 从最内维到最外维逐维 divmod + stride 累加 ----
    #pragma clang loop unroll(full)
    for (int d = (int)OUT_DIM - 1; d >= 0; d--) {
        off_t out_d = (off_t)out_shape[d];

        // TREMS: coord = idx % out_d
        //             退路: 用 TDIVS+TMULS+TSUB 三条拼出取余。
        TREMS(coordTile, idxTile, out_d);

        // TDIVS: idx = idx / out_d (推进到下一维)
        TDIVS(idxTile, idxTile, out_d);

        // 该输出维是否对应输入维
        int i = d - (int)(OUT_DIM - IN_DIM);
        if (i >= 0) {
            if (in_shape[i] != 1) {            // 非广播维才累加
                // TMULS: tmp = coord * stride
                TMULS(tmpTile, coordTile, (off_t)stride);
                // TADD: out += tmp
                TADD(out, out, tmpTile);
            }
            stride *= in_shape[i];             // stride 更新 (广播维 ==1 不变)
        }
    }

    // 不再乘 sizeof(dtype):
    //   PTO ISA MGATHER<Coalesce::Elem> 按"元素索引"取数 (dst[i,j] = src[idx[i,j]])，
    //   而非旧 MGATHER 的字节偏移。
}


// ----------------------------------------------------------------------------
// broadcast: 接口与原 broadcast.hpp 一致
// ----------------------------------------------------------------------------
template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t gIM, size_t gOM, size_t tM>
void broadcast(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape
    ) {
    const size_t Mb = gOM / tM;
    const size_t rmd_M = gOM % tM;

    using gm_shapeIn  = global_tensor<dtype, RowMajor<1, gIM>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using tile_shapeData      = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor>;
    using tile_shapeOffset    = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd  = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeOffset_rmd= Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor, 1, rmd_M>;

    gm_shapeIn inGm(in_ptr);
    tile_shapeData outTile;
    tile_shapeOffset offsetTile;
    tile_shapeData_rmd outTile_rmd;
    tile_shapeOffset_rmd offsetTile_rmd;
    size_t base = 0;

    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
    itOut gOIter(out_ptr);

    size_t total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);

        // 计算偏移 tile (元素索引)
        gen_offset_pto<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>(
            offsetTile, in_shape, out_shape, base, total_elements);
        base += total_elements;

        // MGATHER: 按 offsetTile 中的元素索引从 inGm 取数
        //             但不支持 Coalesce::Elem 模板参数；
        //             且旧实现按字节偏移取数，与 PTO ISA 的元素索引语义不同。
        MGATHER(outTile, inGm, offsetTile);

        // TSTORE: 将 outTile 写回 global memory
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;
        gen_offset_pto<dtype, tile_shapeOffset_rmd, MAX_DIM, IN_DIM, OUT_DIM>(
            offsetTile_rmd, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(outTile_rmd, inGm, offsetTile_rmd);
        TSTORE(gO, outTile_rmd);
    }
}
