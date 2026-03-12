from pto_kernels.registry import included_kernel_records, kernel_counts, load_gap_board


def test_kernel_inventory_counts_match_plan():
    counts = kernel_counts()
    assert counts["included"] == 81
    assert counts["excluded_ai_cpu"] == 2
    assert counts["excluded_a3_only"] == 11
    assert counts["seed_kernels"] == 6


def test_included_kernel_inventory_contains_seed_and_wave_data():
    records = included_kernel_records()
    names = {record.name for record in records}
    assert "apply_rotary_pos_emb" in names
    assert "matmul_reduce_scatter" in names
    assert any(record.wave == "wave5" for record in records if record.name == "matmul_reduce_scatter")


def test_gap_board_is_seed_focused():
    gap_board = load_gap_board()
    assert len(gap_board["gaps"]) >= 3
    assert "flash_attention_score" in gap_board["seed_kernel_scope"]
