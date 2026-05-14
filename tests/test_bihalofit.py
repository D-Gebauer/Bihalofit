"""Regression tests against values from the original C++ ``main.cpp``.

The C++ reference, built with ``g++ -O2 main.cpp bihalofit.cpp`` and run with
the bundled ``linear_pk_planck2015.txt``, prints (at six decimals, z=0.4):

    D1, r_sigma           = 0.809245, 2.010843
    bispec(1.0,1.5,2.0)   = 189401.831773
    bispec_tree           = 1666.239518
    baryon_ratio          = 1.035534
"""
from __future__ import annotations

import numpy as np
import pytest

from bihalofit import BiHalofit, planck2015_pk_path


@pytest.fixture(scope="module")
def bh_tab() -> BiHalofit:
    bh = BiHalofit()
    bh.load_pk_data(planck2015_pk_path())
    return bh


@pytest.fixture(scope="module")
def bh_eh() -> BiHalofit:
    return BiHalofit()


# Reference values from the C++ ``main.cpp`` run.
CPP_REF = dict(
    z=0.4,
    k1=1.0, k2=1.5, k3=2.0,
    D1=0.809245,
    r_sigma=2.010843,
    bispec=189401.831773,
    bispec_tree=1666.239518,
    baryon_ratio=1.035534,
)


def test_D1_matches_cpp(bh_tab):
    assert bh_tab.D1(CPP_REF["z"]) == pytest.approx(CPP_REF["D1"], rel=1e-5)


def test_r_sigma_matches_cpp(bh_tab):
    assert bh_tab.r_sigma(CPP_REF["z"]) == pytest.approx(CPP_REF["r_sigma"], rel=1e-5)


def test_bispec_matches_cpp(bh_tab):
    val = bh_tab.bispec(CPP_REF["k1"], CPP_REF["k2"], CPP_REF["k3"], CPP_REF["z"])
    assert val == pytest.approx(CPP_REF["bispec"], rel=1e-5)


def test_bispec_tree_matches_cpp(bh_tab):
    val = bh_tab.bispec_tree(CPP_REF["k1"], CPP_REF["k2"], CPP_REF["k3"], CPP_REF["z"])
    assert val == pytest.approx(CPP_REF["bispec_tree"], rel=1e-5)


def test_baryon_ratio_matches_cpp(bh_tab):
    val = bh_tab.baryon_ratio(CPP_REF["k1"], CPP_REF["k2"], CPP_REF["k3"], CPP_REF["z"])
    assert val == pytest.approx(CPP_REF["baryon_ratio"], rel=1e-5)


def test_eh_returns_finite_bispectrum(bh_eh):
    val = bh_eh.bispec(1.0, 1.5, 2.0, 0.4)
    assert np.isfinite(val) and val > 0.0


def test_scalar_and_array_agree(bh_tab):
    z = 0.4
    ks = np.array([0.5, 1.0, 2.0, 3.0])
    expected = np.array([bh_tab.bispec(k, k, k, z) for k in ks])
    actual = bh_tab.bispec(ks, ks, ks, z)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_array_broadcasting(bh_tab):
    z = 0.3
    k1 = np.array([1.0, 2.0])
    k2 = np.array([[1.0], [1.5]])
    k3 = 2.0
    out = bh_tab.bispec(k1, k2, k3, z)
    assert out.shape == (2, 2)


def test_growth_factor_normalisation(bh_eh):
    assert bh_eh.D1(0.0) == pytest.approx(1.0, rel=1e-6)


def test_linear_pk_scalar_array(bh_tab):
    ks = np.geomspace(1e-2, 5.0, 16)
    arr = bh_tab.linear_pk(ks)
    scal = np.array([bh_tab.linear_pk(k) for k in ks])
    np.testing.assert_allclose(arr, scal, rtol=1e-12)


def test_z_cache_returns_consistent_values(bh_tab):
    """Repeated z calls should return identical floats (cache hit)."""
    assert bh_tab.D1(0.7) == bh_tab.D1(0.7)
    assert bh_tab.r_sigma(0.7) == bh_tab.r_sigma(0.7)


def test_high_z_uses_tree_level(bh_tab):
    """At z>10 bispec() must coincide with bispec_tree()."""
    val_nl = bh_tab.bispec(1.0, 1.5, 2.0, 12.0)
    val_tree = bh_tab.bispec_tree(1.0, 1.5, 2.0, 12.0)
    assert val_nl == pytest.approx(val_tree, rel=1e-12)
