"""Python entrypoints for PTO kernels and superproject tooling."""

# Import torch when available to avoid
# "libc10.so: cannot open shared object file: No such file or directory".
# See https://github.com/facebookresearch/pytorch3d/issues/1531#issuecomment-1538198217
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

__all__ = ["HAS_EXTENSION"]

try:
    from .pto_kernels_ops import *  # type: ignore # noqa: F401,F403
except ImportError:
    HAS_EXTENSION = False
else:
    HAS_EXTENSION = True
