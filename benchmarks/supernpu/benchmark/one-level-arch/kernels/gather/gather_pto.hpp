// PTO/LinxISA v0.58 gather workload using the canonical TLSU MGATHER path.
#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>

// ----------------------------------------------------------------------------
// gather: 按行索引从数据表中 gather 行
//
// 输入:
//   in_data_ptr   — 数据表, shape (gK, gN), row-major
//   in_offset_ptr — 行索引数组, shape (gM,), 每个元素是数据表的行号
//   out_ptr       — 输出, shape (gM, gN), out[j,:] = in_data[in_offset[j],:]
//
// 模板参数 (与原 gather.hpp 一致):
//   dtype  — 数据类型 (float, half, ...)
//   otype  — 索引类型 (uint32_t, int32_t)
//   gK     — 数据表行数
//   gM     — 输出行数 (= 索引数组长度)
//   gN     — 数据表列数 (= 每行元素数)
//   tM     — tile 行数
//   tN     — tile 列数
// ----------------------------------------------------------------------------
template<typename dtype = float, typename otype = uint32_t,
         size_t gK, size_t gM, size_t gN, size_t tM, size_t tN>
void gather(
    dtype *in_data_ptr,
    otype *in_offset_ptr,
    dtype *out_ptr
    ) {
    const size_t Mb = gM / tM;
    const size_t Nb = gN / tN;
    const size_t rmd_M = gM % tM;
    const size_t rmd_N = gN % tN;

    using gm_shapeInOffset = global_tensor<otype, RowMajor<1, gM>>;
    using gm_shapeIn       = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeOut      = global_tensor<dtype, RowMajor<gM, gN>>;

    using tile_shapeInOffset     = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData         = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor>;
    using tile_shapeInOffset_rmd_n   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor>;
    using tile_shapeData_rmd_n       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, tM, rmd_N>;
    using tile_shapeInOffset_rmd_mn  = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_mn      = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, rmd_N>;
    using tile_shapeInOffset_rmd_m   = Tile<Location::Vec, otype,    1,   tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeData_rmd_m       = Tile<Location::Vec, dtype,    tM,  tN, BLayout::RowMajor, rmd_M, tN>;

    tile_shapeInOffset inOffsetTile;
    tile_shapeData outTile;
    tile_shapeInOffset_rmd_n inOffsetTile_rmd_n;
    tile_shapeData_rmd_n outTile_rmd_n;
    tile_shapeInOffset_rmd_mn inOffsetTile_rmd_mn;
    tile_shapeData_rmd_mn outTile_rmd_mn;
    tile_shapeInOffset_rmd_m inOffsetTile_rmd_m;
    tile_shapeData_rmd_m outTile_rmd_m;

    using itInOffset = global_iterator<gm_shapeInOffset, tile_shapeInOffset>;
    using itOut      = global_iterator<gm_shapeOut, tile_shapeData>;

    itInOffset gInOffsetIter(in_offset_ptr);
    itOut      gOIter(out_ptr);

    // ---- 主循环: Mb × Nb 个完整 tile ----
    for (int j = 0; j < Mb; ++j) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, i);
            size_t n_base  = i * tN;

            // TLOAD: 加载行索引 tile (1, tM) from GM
            TLOAD(inOffsetTile, gInOffset);

            // MGATHER<Coalesce::Row>: 按行索引从数据表取数
            //   dst[r,:] = table[idx[r], :]
            //   table 指针偏移 n_base 个元素, 使取数起始列为 n_base
            //   (tablePtr + idx * gN + n_base 定位到 row idx, col n_base)
            //             且按字节偏移取数, 非行索引
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile, adjustedGm, inOffsetTile);

            // TSTORE: 写回输出 tile (tM, tN) to GM
            TSTORE(gO, outTile);
        }

        // ---- rmd_N: 最后一个列块不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, j);
            auto gO        = gOIter(j, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_n, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_n, adjustedGm, inOffsetTile_rmd_n);
            TSTORE(gO, outTile_rmd_n);
        }
    }

    // ---- rmd_M: 最后一个行块不完整 ----
    if constexpr (rmd_M) {
        for (int i = 0; i < Nb; ++i) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, i);
            size_t n_base  = i * tN;

            TLOAD(inOffsetTile_rmd_m, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_m, adjustedGm, inOffsetTile_rmd_m);
            TSTORE(gO, outTile_rmd_m);
        }

        // ---- rmd_M + rmd_N: 右下角不完整 ----
        if constexpr (rmd_N) {
            auto gInOffset = gInOffsetIter(0, Mb);
            auto gO        = gOIter(Mb, Nb);
            size_t n_base  = Nb * tN;

            TLOAD(inOffsetTile_rmd_mn, gInOffset);
            gm_shapeIn adjustedGm(in_data_ptr + n_base);
            MGATHER(outTile_rmd_mn, adjustedGm, inOffsetTile_rmd_mn);
            TSTORE(gO, outTile_rmd_mn);
        }
    }
}
