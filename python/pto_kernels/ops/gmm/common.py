"""Shared GMM scheduling helpers shaped after the PTODSL matmul optimization examples."""

from ptodsl import pto


const = pto.const


def swizzle_zn(logical_block, m_loop, n_loop, swizzle_count):
    """Row-block swizzle matching the local PTODSL matmul_swizzle example."""
    c1 = const(1)
    c2 = const(2)
    tile_block_loop = pto.ceil_div(m_loop, swizzle_count)
    tile_block_span = swizzle_count * n_loop
    tile_block_idx = logical_block // tile_block_span
    in_tile_block_idx = logical_block % tile_block_span
    is_last_block = tile_block_idx == (tile_block_loop - c1)
    n_row_tail = m_loop - swizzle_count * tile_block_idx
    n_row = pto.select(is_last_block, n_row_tail, swizzle_count)
    m_idx = tile_block_idx * swizzle_count + (in_tile_block_idx % n_row)
    n_idx = in_tile_block_idx // n_row
    odd_block = (tile_block_idx % c2) == c1
    flipped_n_idx = n_loop - n_idx - c1
    n_idx = pto.select(odd_block, flipped_n_idx, n_idx)
    return m_idx, n_idx


def swizzle_nz(logical_block, m_loop, n_loop, swizzle_count):
    """Column-block swizzle matching the local PTODSL matmul_swizzle example."""
    c1 = const(1)
    c2 = const(2)
    tile_block_loop = pto.ceil_div(n_loop, swizzle_count)
    tile_block_span = swizzle_count * m_loop
    tile_block_idx = logical_block // tile_block_span
    in_tile_block_idx = logical_block % tile_block_span
    is_last_block = tile_block_idx == (tile_block_loop - c1)
    n_col_tail = n_loop - swizzle_count * tile_block_idx
    n_col = pto.select(is_last_block, n_col_tail, swizzle_count)
    m_idx = in_tile_block_idx // n_col
    n_idx = tile_block_idx * swizzle_count + (in_tile_block_idx % n_col)
    odd_block = (tile_block_idx % c2) == c1
    flipped_m_idx = m_loop - m_idx - c1
    m_idx = pto.select(odd_block, flipped_m_idx, m_idx)
    return m_idx, n_idx
