# bihalofit (Python)

A Python package implementing **BiHalofit** — the fitting formula of the
non-linear matter bispectrum from Takahashi et al. 2019
([arXiv:1911.07886](https://arxiv.org/abs/1911.07886)).

This is a port of the C++ implementation that lives in `../bihalofit_cpp/`,
with the numerically intensive routines (`sigma(r)`, the RK4 growth-factor
integrator, `r_sigma` bisection, the bispectrum kernel) compiled to native
code via [Numba](https://numba.pydata.org/). Vectorized calls are parallelized
across CPU cores. Per-redshift quantities are cached so a batch of triangles
at the same `z` only pays for `D1(z)`, `r_sigma(z)`, and `n_eff(z)` once.

## Installation

```bash
cd bihalofit_python
pip install -e .
```

Dependencies: `numpy`, `numba`. The first import of the package will JIT-compile
the kernels (a few seconds, one-off — cached to disk via `cache=True`).

## Python API

```python
import numpy as np
from bihalofit import BiHalofit, planck2015_pk_path

# Default constructor: Planck 2015 flat-LCDM cosmology, Eisenstein-Hu P(k).
bh = BiHalofit()

# Optionally load a tabulated linear P(k) at z=0 (overrides EH formula):
bh.load_pk_data(planck2015_pk_path())
# Or directly from arrays:
# bh.set_pk_data(k_array, pk_array)

# Custom flat wCDM cosmology:
# bh = BiHalofit(omb=0.047, omc=0.286, h=0.7, sigma8=0.82, ns=0.96, w=-1.0)

z = 0.4
print("D1(z)       =", bh.D1(z))
print("r_sigma(z)  =", bh.r_sigma(z), "Mpc/h")
print("n_eff(z)    =", bh.n_eff(z))

# Scalar bispectra:
print(bh.bispec(1.0, 1.5, 2.0, z))         # non-linear, no baryons
print(bh.bispec_tree(1.0, 1.5, 2.0, z))    # tree-level
print(bh.baryon_ratio(1.0, 1.5, 2.0, z))   # baryon-ratio fit
print(bh.bispec_with_baryons(1.0, 1.5, 2.0, z))

# Vectorized: any broadcastable combination of k1, k2, k3 works.
ks = np.geomspace(0.1, 5.0, 200)
b_eq = bh.bispec(ks, ks, ks, z)            # equilateral
b_iso = bh.bispec(ks, ks, 2*ks, z)         # isoceles
```

## Command-line interface

The package installs a `bihalofit` executable. It mirrors the workflow of the
original `main.cpp`:

```bash
# Single triangle, bundled Planck 2015 P(k):
bihalofit --pk planck2015 -k 1.0 1.5 2.0 -z 0.4

# Multiple triangles, multiple redshifts, tab-separated with header:
bihalofit --pk planck2015 --header \
          -k 0.1 0.1 0.1 -k 1 1 1 -k 5 5 5 \
          -z 0.0 -z 0.5 -z 1.0

# Custom cosmology, EH P(k), CSV output:
bihalofit --omb 0.047 --omc 0.286 --h 0.7 --sigma8 0.82 --ns 0.96 --w -1 \
          --sep , -k 1 1.5 2 -z 0.4
```

Columns: `z`, `k1`, `k2`, `k3`, `D1`, `r_sigma`, `bispec`, `bispec_tree`,
`baryon_ratio`. Pass `--no-baryons` to drop the baryon-ratio column.

Run `bihalofit --help` for the full option list.

## Equivalence with the C++ code

The Python kernels were translated line-for-line from `bihalofit.cpp`. Outputs
agree with the C++ implementation to the convergence tolerance of the
underlying numerical schemes (relative `EPS = 1e-4` for `sigma(r)`, the RK4
growth-factor integral, and the `r_sigma` bisection).
