from pto_kernels.bench.adapter_utils import blocked, describe_baseline


def describe(repo_root, spec):
    return describe_baseline(repo_root, "moe", "moe_token_permute", spec.inventory_ref)


def compile_kernel(repo_root, spec, artifacts_dir):
    return blocked("Capture ops-transformer compile/run flow for moe_token_permute.")


def benchmark(repo_root, spec, artifacts_dir):
    return blocked("Benchmark capture is pending baseline adapter implementation.")
