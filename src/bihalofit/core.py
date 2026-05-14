"""High-level Pythonic interface to BiHalofit.

The :class:`BiHalofit` class mirrors the C++ ``bihalofit`` object: instantiate
it with cosmological parameters (or use the Planck 2015 defaults), optionally
hand it a tabulated linear P(k), then call :meth:`bispec`, :meth:`bispec_tree`,
:meth:`baryon_ratio`, :meth:`D1`, :meth:`r_sigma`, or :meth:`linear_pk`.

All bispectrum methods accept either scalar floats or NumPy arrays for
``k1, k2, k3``. Array inputs are broadcast together and computed in parallel
through Numba-compiled kernels. Expensive per-redshift quantities (the linear
growth factor ``D1(z)`` and the non-linear scale ``r_sigma(z)``) are cached
keyed by ``z`` so they are computed once per redshift across an entire batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from . import _kernels as K

ArrayLike = Union[float, np.ndarray]

# Planck 2015 flat-LambdaCDM defaults, matching bihalofit.cpp.
DEFAULTS = dict(omb=0.0492, omc=0.2664, h=0.6727, sigma8=0.831, ns=0.9645, w=-1.0)


@dataclass
class _ZCache:
    D1: float
    r_sigma: float
    n_eff: float


class BiHalofit:
    """BiHalofit non-linear matter bispectrum (Takahashi+ 2019).

    Parameters
    ----------
    omb, omc : float
        Baryon and CDM density parameters at z=0.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
    sigma8 : float
        Linear sigma8.
    ns : float
        Spectral index of the linear power spectrum.
    w : float
        Constant equation-of-state of dark energy (flat wCDM).
    pk_table : tuple(ndarray, ndarray), optional
        ``(k, P(k))`` tabulated linear power spectrum at z=0. If omitted, the
        Eisenstein & Hu (1999) no-wiggle fitting formula is used.
    """

    def __init__(
        self,
        omb: float = DEFAULTS["omb"],
        omc: float = DEFAULTS["omc"],
        h: float = DEFAULTS["h"],
        sigma8: float = DEFAULTS["sigma8"],
        ns: float = DEFAULTS["ns"],
        w: float = DEFAULTS["w"],
        pk_table: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.omb = float(omb)
        self.omc = float(omc)
        self.h = float(h)
        self.sigma8 = float(sigma8)
        self.ns = float(ns)
        self.w = float(w)

        self.om = self.omb + self.omc
        self.ow = 1.0 - self.om

        # Storage for the optional tabulated P(k). Always allocate (1-element
        # dummy when unused) so Numba sees a well-typed array.
        self._k_data = np.zeros(1, dtype=np.float64)
        self._pk_data = np.zeros(1, dtype=np.float64)
        self._use_data = False

        if pk_table is not None:
            self.set_pk_data(*pk_table)

        self.norm = 1.0  # will be replaced by compute_pk_norm
        self._z_cache: dict[float, _ZCache] = {}
        self._compute_pk_norm()

    # ------------------------------------------------------------------
    # Power spectrum table handling
    # ------------------------------------------------------------------
    def set_pk_data(self, k: np.ndarray, pk: np.ndarray) -> None:
        """Install a tabulated linear P(k) at z=0 (k in h/Mpc, P in (Mpc/h)^3)."""
        k = np.ascontiguousarray(k, dtype=np.float64)
        pk = np.ascontiguousarray(pk, dtype=np.float64)
        if k.shape != pk.shape or k.ndim != 1:
            raise ValueError("k and pk must be 1-D arrays of the same shape")
        order = np.argsort(k)
        self._k_data = k[order].copy()
        self._pk_data = pk[order].copy()
        self._use_data = True
        self._z_cache.clear()
        self._compute_pk_norm()

    def load_pk_data(self, filename: Union[str, Path]) -> None:
        """Load a two-column ASCII table (k, P(k))."""
        data = np.loadtxt(filename)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{filename}: expected two-column ASCII table")
        self.set_pk_data(data[:, 0], data[:, 1])

    def use_eh(self) -> None:
        """Switch back to the Eisenstein & Hu (1999) no-wiggle P(k)."""
        self._use_data = False
        self._z_cache.clear()
        self._compute_pk_norm()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_pk_norm(self) -> None:
        """Normalize P(k) amplitude so sigma_8 matches the requested value."""
        # Bootstrap with norm=1, evaluate sigma_8 from the formula/table.
        self.norm = 1.0
        s8 = K.sigmam(
            8.0, 0, self.h, self.om, self.omc, self.omb, self.ns, self.norm,
            self._k_data, self._pk_data, self._use_data,
        )
        self.norm = self.sigma8 / s8

    def _ensure_z(self, z: float) -> _ZCache:
        cached = self._z_cache.get(z)
        if cached is not None:
            return cached
        D1 = K.calc_D1(z, self.om, self.ow, self.w)
        r_sigma = K.calc_r_sigma(
            z, D1, self.h, self.om, self.omc, self.omb, self.ns, self.norm,
            self.ow, self.w, self._k_data, self._pk_data, self._use_data,
        )
        s_rsig = K.sigmam(
            r_sigma, 2, self.h, self.om, self.omc, self.omb, self.ns, self.norm,
            self._k_data, self._pk_data, self._use_data,
        )
        n_eff = -3.0 + 2.0 * (D1 * s_rsig) ** 2
        cache = _ZCache(D1=D1, r_sigma=r_sigma, n_eff=n_eff)
        self._z_cache[z] = cache
        return cache

    def _pk_args(self):
        return (self.h, self.om, self.omc, self.omb, self.ns, self.norm,
                self._k_data, self._pk_data, self._use_data)

    # ------------------------------------------------------------------
    # Linear quantities
    # ------------------------------------------------------------------
    def linear_pk(self, k: ArrayLike) -> ArrayLike:
        """Linear matter P(k) at z=0 (h/Mpc, (Mpc/h)^3)."""
        karr = np.asarray(k, dtype=np.float64)
        if karr.ndim == 0:
            return float(K.linear_pk(karr.item(), *self._pk_args()))
        out = K.linear_pk_array(np.ascontiguousarray(karr.ravel()),
                                *self._pk_args())
        return out.reshape(karr.shape)

    def D1(self, z: float) -> float:
        """Linear growth factor at z, normalized to D1(0)=1."""
        return float(self._ensure_z(float(z)).D1)

    def r_sigma(self, z: float) -> float:
        """Non-linear scale r_sigma(z) in Mpc/h (Eq. B1)."""
        return float(self._ensure_z(float(z)).r_sigma)

    def n_eff(self, z: float) -> float:
        """Effective spectral index n_eff(z) at r_sigma (Eq. B2)."""
        return float(self._ensure_z(float(z)).n_eff)

    # ------------------------------------------------------------------
    # Bispectra
    # ------------------------------------------------------------------
    def bispec_tree(self, k1: ArrayLike, k2: ArrayLike, k3: ArrayLike,
                    z: float) -> ArrayLike:
        """Tree-level bispectrum [(Mpc/h)^6]."""
        c = self._ensure_z(float(z))
        k1a, k2a, k3a, shape = _broadcast_triple(k1, k2, k3)
        if shape == ():
            return float(K.bispec_tree_precomp(
                k1a[0], k2a[0], k3a[0], c.D1, *self._pk_args()))
        out = K.bispec_tree_array(k1a, k2a, k3a, c.D1, *self._pk_args())
        return out.reshape(shape)

    def bispec(self, k1: ArrayLike, k2: ArrayLike, k3: ArrayLike,
               z: float) -> ArrayLike:
        """Non-linear bispectrum without baryons [(Mpc/h)^6]."""
        z = float(z)
        c = self._ensure_z(z)
        k1a, k2a, k3a, shape = _broadcast_triple(k1, k2, k3)
        if shape == ():
            return float(K.bispec_precomp(
                k1a[0], k2a[0], k3a[0], z,
                c.D1, c.r_sigma, c.n_eff,
                self.sigma8, self.om, self.ow, self.w,
                self.h, self.omc, self.omb, self.ns, self.norm,
                self._k_data, self._pk_data, self._use_data,
            ))
        out = K.bispec_array(
            k1a, k2a, k3a, z,
            c.D1, c.r_sigma, c.n_eff,
            self.sigma8, self.om, self.ow, self.w,
            self.h, self.omc, self.omb, self.ns, self.norm,
            self._k_data, self._pk_data, self._use_data,
        )
        return out.reshape(shape)

    def baryon_ratio(self, k1: ArrayLike, k2: ArrayLike, k3: ArrayLike,
                     z: float) -> ArrayLike:
        """Bispectrum ratio with/without baryons (Eq. C1)."""
        z = float(z)
        k1a, k2a, k3a, shape = _broadcast_triple(k1, k2, k3)
        if shape == ():
            return float(K.baryon_ratio(k1a[0], k2a[0], k3a[0], z))
        out = K.baryon_ratio_array(k1a, k2a, k3a, z)
        return out.reshape(shape)

    def bispec_with_baryons(self, k1: ArrayLike, k2: ArrayLike, k3: ArrayLike,
                            z: float) -> ArrayLike:
        """Non-linear bispectrum multiplied by the baryon-ratio fit."""
        return self.bispec(k1, k2, k3, z) * self.baryon_ratio(k1, k2, k3, z)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        src = "table" if self._use_data else "Eisenstein-Hu"
        return (
            f"BiHalofit(omb={self.omb:g}, omc={self.omc:g}, h={self.h:g}, "
            f"sigma8={self.sigma8:g}, ns={self.ns:g}, w={self.w:g}, "
            f"linear_pk={src})"
        )


def _broadcast_triple(k1, k2, k3):
    a = np.broadcast_arrays(
        np.asarray(k1, dtype=np.float64),
        np.asarray(k2, dtype=np.float64),
        np.asarray(k3, dtype=np.float64),
    )
    shape = a[0].shape
    flat = [np.ascontiguousarray(np.atleast_1d(x).ravel()) for x in a]
    return flat[0], flat[1], flat[2], shape


def planck2015_pk_path() -> Path:
    """Return the path to the bundled Planck 2015 linear P(k) table."""
    return Path(__file__).parent / "data" / "linear_pk_planck2015.txt"
