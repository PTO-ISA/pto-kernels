from pto_kernels.bench.adapter_utils import blocked, compile_pto_kernel, describe_pto


KERNEL = "python/pto_kernels/ops/posembedding/apply_rotary_pos_emb/kernel.py"
META = "python/pto_kernels/ops/posembedding/apply_rotary_pos_emb/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    return blocked("PTO benchmark is pending kernel port implementation.")
