"""Python entrypoints for PTO kernels and superproject tooling."""

# Import torch before the extension so its shared libraries, including libc10,
# are available when pto_kernels_ops is loaded.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

try:
    from .benchmarking import do_bench  # noqa: F401
except ImportError:
    do_bench = None

__all__ = ["HAS_EXTENSION"]
if do_bench is not None:
    __all__.append("do_bench")

try:
    from .pto_kernels_ops import *  # type: ignore # noqa: F401,F403
except ImportError:
    HAS_EXTENSION = False
else:
    HAS_EXTENSION = True
