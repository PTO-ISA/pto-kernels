from pto_kernels.bench.adapter_utils import blocked, describe_baseline


def describe(repo_root, spec):
    return describe_baseline(repo_root, "attention", "flash_attention_score", spec.inventory_ref)


def compile_kernel(repo_root, spec, artifacts_dir):
    return blocked("Capture ops-transformer compile/run flow for flash_attention_score.")


def benchmark(repo_root, spec, artifacts_dir):
    return blocked("Benchmark capture is pending baseline adapter implementation.")
