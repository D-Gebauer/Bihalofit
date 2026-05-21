"""BiHalofit: non-linear matter bispectrum (Takahashi+ 2019, arXiv:1911.07886).

A Python port of the original C++ BiHalofit code, with Numba-JIT compiled
numerical kernels for native performance.

Quick start
-----------
>>> from bihalofit import BiHalofit
>>> bh = BiHalofit()                       # Planck 2015 defaults, EH P(k)
>>> bh.load_pk_data("linear_pk.txt")       # optional tabulated P(k)
>>> bh.bispec(1.0, 1.5, 2.0, z=0.4)        # scalar call
>>> import numpy as np
>>> ks = np.linspace(0.5, 3.0, 64)
>>> bh.bispec(ks, ks, ks, z=0.4)           # vectorized (parallelized) call
"""
from __future__ import annotations

import os as _os

# Numba's parallel kernels can crash with a recursive `std::terminate` /
# `pthread_create … Resource temporarily unavailable` when two OpenMP runtimes
# end up loaded in the same process (e.g. NumPy's libiomp5md.dll alongside
# Numba's libgomp on Anaconda for Windows). Defaulting to the `workqueue`
# threading layer avoids that conflict entirely. Power users can override by
# setting BIHALOFIT_NUMBA_LAYER (or the standard NUMBA_THREADING_LAYER).
_os.environ.setdefault(
    "NUMBA_THREADING_LAYER",
    _os.environ.get("BIHALOFIT_NUMBA_LAYER", "workqueue"),
)

from .core import BiHalofit, DEFAULTS, planck2015_pk_path  # noqa: E402

__all__ = ["BiHalofit", "DEFAULTS", "planck2015_pk_path"]
__version__ = "0.1.0"
