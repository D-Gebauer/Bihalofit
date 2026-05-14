"""Numba JIT-compiled numerical kernels for BiHalofit.

These are translated as faithfully as possible from the original C++
implementation (Takahashi et al. 2019, arXiv:1911.07886). All kernels are
``@njit`` so they compile to native code on first invocation; results are
cached on disk via ``cache=True`` so subsequent process starts are fast.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

EPS = 1.0e-4  # fractional accuracy of BS computation (matches C++)


# ---------------------------------------------------------------------------
# Linear power spectrum
# ---------------------------------------------------------------------------

@njit(cache=True)
def linear_pk_eh(k_in, h, om, omc, omb, ns, norm):
    """Eisenstein & Hu (1999) no-wiggle linear P(k). k in h/Mpc."""
    k = k_in * h  # convert h/Mpc to 1/Mpc

    fc = omc / om
    fb = omb / om
    theta = 2.728 / 2.7
    pc = 0.25 * (5.0 - np.sqrt(1.0 + 24.0 * fc))
    omh2 = om * h * h
    ombh2 = omb * h * h
    zeq = 2.5e4 * omh2 / theta**4
    b1 = 0.313 * omh2**(-0.419) * (1.0 + 0.607 * omh2**0.674)
    b2 = 0.238 * omh2**0.223
    zd = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * ombh2**b2)
    yd = (1.0 + zeq) / (1.0 + zd)
    sh = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)

    sqrt_alnu_term = fc * (5.0 - 2.0 * pc) / 5.0 * (1.0 - 0.553 * fb + 0.126 * fb**3)
    alnu = (
        sqrt_alnu_term
        * (1.0 + yd)**(-pc)
        * (1.0 + 0.5 * pc * (1.0 + 1.0 / (7.0 * (3.0 - 4.0 * pc))) / (1.0 + yd))
    )

    sqrt_alnu = np.sqrt(alnu)
    geff = omh2 * (sqrt_alnu + (1.0 - sqrt_alnu) / (1.0 + (0.43 * k * sh)**4))
    qeff = k / geff * theta * theta

    L = np.log(np.e + 1.84 * sqrt_alnu * qeff / (1.0 - 0.949 * fb))
    C = 14.4 + 325.0 / (1.0 + 60.5 * qeff**1.11)

    delk = norm * norm * (k * 2997.9 / h)**(3.0 + ns) * (L / (L + C * qeff * qeff))**2
    pk = 2.0 * np.pi * np.pi / (k * k * k) * delk

    return h**3 * pk


@njit(cache=True)
def linear_pk_data(k, k_data, pk_data, norm):
    """Log-log interpolation of a tabulated linear P(k)."""
    n_data = k_data.shape[0]
    if k < k_data[0]:
        return 0.0
    if k > k_data[n_data - 1]:
        return 0.0

    lk = np.log10(k)
    j1 = 0
    j2 = n_data - 1
    while j2 - j1 > 1:
        jm = (j1 + j2) // 2
        if k > k_data[jm]:
            j1 = jm
        else:
            j2 = jm
    j = j1

    lk_j = np.log10(k_data[j])
    lk_j1 = np.log10(k_data[j + 1])
    lp_j = np.log10(pk_data[j])
    lp_j1 = np.log10(pk_data[j + 1])
    f = (lp_j1 - lp_j) / (lk_j1 - lk_j) * (lk - lk_j) + lp_j
    return norm * norm * 10.0**f


@njit(cache=True)
def linear_pk(k, h, om, omc, omb, ns, norm, k_data, pk_data, use_data):
    if use_data:
        return linear_pk_data(k, k_data, pk_data, norm)
    return linear_pk_eh(k, h, om, omc, omb, ns, norm)


# ---------------------------------------------------------------------------
# Windows, sigma(r), growth factor
# ---------------------------------------------------------------------------

@njit(cache=True)
def window(x, i):
    if i == 0:  # top-hat
        return 3.0 / (x * x * x) * (np.sin(x) - x * np.cos(x))
    if i == 1:  # gaussian
        return np.exp(-0.5 * x * x)
    # i == 2: first derivative of gaussian (scaled)
    return x * np.exp(-0.5 * x * x)


@njit(cache=True)
def sigmam(r, j, h, om, omc, omb, ns, norm, k_data, pk_data, use_data):
    """sigma(r) computed via iterated trapezoidal rule in log-k.

    Mirrors the adaptive C++ implementation: doubles the integration range
    outward and refines the grid until convergence to EPS.
    """
    k1 = 2.0 * np.pi / r
    k2 = 2.0 * np.pi / r

    xxpp = -1.0
    xx = 0.0
    while True:
        k1 = k1 / 10.0
        k2 = k2 * 2.0
        a = np.log(k1)
        b = np.log(k2)

        xxp = -1.0
        n = 2
        while True:
            n = n * 2
            hh = (b - a) / n
            xx = 0.0
            for i in range(1, n):
                kk = np.exp(a + hh * i)
                ww = window(kk * r, j)
                pk = linear_pk(kk, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
                xx += kk * kk * kk * pk * ww * ww
            w1 = window(k1 * r, j)
            w2 = window(k2 * r, j)
            pk1 = linear_pk(k1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
            pk2 = linear_pk(k2, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
            xx += 0.5 * (k1**3 * pk1 * w1 * w1 + k2**3 * pk2 * w2 * w2)
            xx *= hh

            if abs((xx - xxp) / xx) < EPS:
                break
            xxp = xx

        if abs((xx - xxpp) / xx) < EPS:
            break
        xxpp = xx

    return np.sqrt(xx / (2.0 * np.pi * np.pi))


@njit(cache=True)
def _lgr_rhs(la, y0, y1, om, ow, w):
    """RHS for the growth-factor ODE in log-a; returns (dy0/dla, dy1/dla)."""
    a = np.exp(la)
    pw = a**(-3.0 * w)
    denom = om + ow * pw
    dy0 = y1
    dy1 = (-0.5 * (5.0 * om + (5.0 - 3.0 * w) * ow * pw) * y1
           - 1.5 * (1.0 - w) * ow * pw * y0) / denom
    return dy0, dy1


@njit(cache=True)
def lgr(z, om, ow, w):
    """Unnormalized linear growth factor at z, via RK4 with grid refinement."""
    a = 1.0 / (1.0 + z)
    a0 = 1.0 / 1100.0

    yp = -1.0
    n = 10
    y0 = 1.0
    while True:
        n *= 2
        hh = (np.log(a) - np.log(a0)) / n
        x = np.log(a0)
        y0 = 1.0
        y1 = 0.0
        for _ in range(n):
            k1_0, k1_1 = _lgr_rhs(x, y0, y1, om, ow, w)
            k1_0 *= hh; k1_1 *= hh
            k2_0, k2_1 = _lgr_rhs(x + 0.5 * hh, y0 + 0.5 * k1_0, y1 + 0.5 * k1_1, om, ow, w)
            k2_0 *= hh; k2_1 *= hh
            k3_0, k3_1 = _lgr_rhs(x + 0.5 * hh, y0 + 0.5 * k2_0, y1 + 0.5 * k2_1, om, ow, w)
            k3_0 *= hh; k3_1 *= hh
            k4_0, k4_1 = _lgr_rhs(x + hh, y0 + k3_0, y1 + k3_1, om, ow, w)
            k4_0 *= hh; k4_1 *= hh

            y0 += (k1_0 + k4_0) / 6.0 + (k2_0 + k3_0) / 3.0
            y1 += (k1_1 + k4_1) / 6.0 + (k2_1 + k3_1) / 3.0
            x += hh

        if abs(y0 / yp - 1.0) < 0.1 * EPS:
            break
        yp = y0

    return a * y0


@njit(cache=True)
def calc_D1(z, om, ow, w):
    return lgr(z, om, ow, w) / lgr(0.0, om, ow, w)


@njit(cache=True)
def calc_r_sigma(z, D1, h, om, omc, omb, ns, norm, ow, w, k_data, pk_data, use_data):
    """Solve D1 * sigma(1/k, gaussian) = 1 via bracket + bisection."""
    k1 = 1.0
    k2 = 1.0
    while True:
        s = D1 * sigmam(1.0 / k1, 1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
        if s < 1.0:
            break
        k1 *= 0.5
    while True:
        s = D1 * sigmam(1.0 / k2, 1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
        if s > 1.0:
            break
        k2 *= 2.0

    k = 0.5 * (k1 + k2)
    while True:
        k = 0.5 * (k1 + k2)
        s = D1 * sigmam(1.0 / k, 1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
        if s < 1.0:
            k1 = k
        elif s > 1.0:
            k2 = k
        if s == 1.0 or abs(k2 / k1 - 1.0) < EPS * 0.1:
            break
    return 1.0 / k


# ---------------------------------------------------------------------------
# Bispectrum kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def F2_tree(k1, k2, k3):
    costheta12 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2) / (k1 * k2)
    return (5.0 / 7.0) + 0.5 * costheta12 * (k1 / k2 + k2 / k1) + (2.0 / 7.0) * costheta12 * costheta12


@njit(cache=True)
def F2_nl(k1, k2, k3, z, D1, r_sigma, sigma8, om, ow, w):
    q3 = k3 * r_sigma
    logsigma8z = np.log10(D1 * sigma8)
    a = 1.0 / (1.0 + z)
    omz = om / (om + ow * a**(-3.0 * w))
    dn = 10.0**(-0.483 + 0.892 * logsigma8z - 0.086 * omz)
    return F2_tree(k1, k2, k3) + dn * q3


@njit(cache=True)
def bispec_tree(k1, k2, k3, z, om, ow, w, h, omc, omb, ns, norm, k_data, pk_data, use_data):
    D1 = calc_D1(z, om, ow, w)
    pk1 = linear_pk(k1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk2 = linear_pk(k2, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk3 = linear_pk(k3, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    return D1**4 * 2.0 * (
        F2_tree(k1, k2, k3) * pk1 * pk2
        + F2_tree(k2, k3, k1) * pk2 * pk3
        + F2_tree(k3, k1, k2) * pk3 * pk1
    )


@njit(cache=True)
def bispec_tree_precomp(k1, k2, k3, D1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data):
    """Tree-level bispectrum given precomputed D1."""
    pk1 = linear_pk(k1, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk2 = linear_pk(k2, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk3 = linear_pk(k3, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    return D1**4 * 2.0 * (
        F2_tree(k1, k2, k3) * pk1 * pk2
        + F2_tree(k2, k3, k1) * pk2 * pk3
        + F2_tree(k3, k1, k2) * pk3 * pk1
    )


@njit(cache=True)
def bispec_precomp(k1, k2, k3, z, D1, r_sigma, n_eff,
                   sigma8, om, ow, w, h, omc, omb, ns, norm,
                   k_data, pk_data, use_data):
    """Non-linear bispectrum given precomputed D1, r_sigma, n_eff."""
    if z > 10.0:
        return bispec_tree_precomp(k1, k2, k3, D1, h, om, omc, omb, ns, norm,
                                   k_data, pk_data, use_data)

    q1 = k1 * r_sigma
    q2 = k2 * r_sigma
    q3 = k3 * r_sigma

    # sort (ka, kb, kc) descending
    ka, kb, kc = k1, k2, k3
    if ka < kb:
        ka, kb = kb, ka
    if ka < kc:
        ka, kc = kc, ka
    if kb < kc:
        kb, kc = kc, kb

    r1 = kc / ka
    r2 = (kb + kc - ka) / ka

    logsigma8z = np.log10(D1 * sigma8)

    # 1-halo term parameters (Eq. B7)
    an = 10.0**(
        -2.167 - 2.944 * logsigma8z
        - 1.106 * logsigma8z**2 - 2.865 * logsigma8z**3
        - 0.310 * r1**(10.0**(0.182 + 0.57 * n_eff))
    )
    bn = 10.0**(-3.428 - 2.681 * logsigma8z + 1.624 * logsigma8z**2 - 0.095 * logsigma8z**3)
    cn = 10.0**(0.159 - 1.107 * n_eff)
    alphan = 10.0**(-4.348 - 3.006 * n_eff - 0.5745 * n_eff**2
                    + 10.0**(-0.9 + 0.2 * n_eff) * r2 * r2)
    cap = 1.0 - (2.0 / 3.0) * ns
    if alphan > cap:
        alphan = cap
    betan = 10.0**(-1.731 - 2.845 * n_eff - 1.4995 * n_eff**2
                   - 0.2811 * n_eff**3 + 0.007 * r2)

    BS1h = 1.0
    BS1h *= 1.0 / (an * q1**alphan + bn * q1**betan) / (1.0 + 1.0 / (cn * q1))
    BS1h *= 1.0 / (an * q2**alphan + bn * q2**betan) / (1.0 + 1.0 / (cn * q2))
    BS1h *= 1.0 / (an * q3**alphan + bn * q3**betan) / (1.0 + 1.0 / (cn * q3))

    # 3-halo term parameters (Eq. B9)
    fn = 10.0**(-10.533 - 16.838 * n_eff - 9.3048 * n_eff**2 - 1.8263 * n_eff**3)
    gn = 10.0**(2.787 + 2.405 * n_eff + 0.4577 * n_eff**2)
    hn = 10.0**(-1.118 - 0.394 * n_eff)
    mn = 10.0**(-2.605 - 2.434 * logsigma8z + 5.71 * logsigma8z**2)
    nn = 10.0**(-4.468 - 3.08 * logsigma8z + 1.035 * logsigma8z**2)
    mun = 10.0**(15.312 + 22.977 * n_eff + 10.9579 * n_eff**2 + 1.6586 * n_eff**3)
    nun = 10.0**(1.347 + 1.246 * n_eff + 0.4525 * n_eff**2)
    pn = 10.0**(0.071 - 0.433 * n_eff)
    en = 10.0**(-0.632 + 0.646 * n_eff)

    pk_q1 = linear_pk(q1 / r_sigma, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk_q2 = linear_pk(q2 / r_sigma, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)
    pk_q3 = linear_pk(q3 / r_sigma, h, om, omc, omb, ns, norm, k_data, pk_data, use_data)

    PSE1 = ((1.0 + fn * q1 * q1) / (1.0 + gn * q1 + hn * q1 * q1) * D1 * D1 * pk_q1
            + 1.0 / (mn * q1**mun + nn * q1**nun) / (1.0 + (pn * q1)**(-3)))
    PSE2 = ((1.0 + fn * q2 * q2) / (1.0 + gn * q2 + hn * q2 * q2) * D1 * D1 * pk_q2
            + 1.0 / (mn * q2**mun + nn * q2**nun) / (1.0 + (pn * q2)**(-3)))
    PSE3 = ((1.0 + fn * q3 * q3) / (1.0 + gn * q3 + hn * q3 * q3) * D1 * D1 * pk_q3
            + 1.0 / (mn * q3**mun + nn * q3**nun) / (1.0 + (pn * q3)**(-3)))

    F2_12 = F2_nl(k1, k2, k3, z, D1, r_sigma, sigma8, om, ow, w)
    F2_23 = F2_nl(k2, k3, k1, z, D1, r_sigma, sigma8, om, ow, w)
    F2_31 = F2_nl(k3, k1, k2, z, D1, r_sigma, sigma8, om, ow, w)

    BS3h = 2.0 * (F2_12 * PSE1 * PSE2 + F2_23 * PSE2 * PSE3 + F2_31 * PSE3 * PSE1)
    BS3h *= 1.0 / (1.0 + en * q1) / (1.0 + en * q2) / (1.0 + en * q3)

    return BS1h + BS3h


@njit(cache=True)
def baryon_ratio(k1, k2, k3, z):
    """Bispectrum ratio with/without baryons (Eq. C1)."""
    if z > 5.0:
        return 1.0

    a = 1.0 / (1.0 + z)

    if a > 0.5:
        A0 = 0.068 * (a - 0.5)**0.47
    else:
        A0 = 0.0
    mu0 = 0.018 * a + 0.837 * a * a
    sigma0 = 0.881 * mu0
    alpha0 = 2.346

    if a > 0.2:
        A1 = 1.052 * (a - 0.2)**1.41
    else:
        A1 = 0.0
    mu1 = abs(0.172 + 3.048 * a - 0.675 * a * a)
    sigma1 = (0.494 - 0.039 * a) * mu1

    ks = 29.90 - 38.73 * a + 24.30 * a * a
    alpha2 = 2.25
    beta2 = 0.563 / ((a / 0.060)**0.02 + 1.0) / alpha2

    Rb = 1.0
    for ki in (k1, k2, k3):
        xi = np.log10(ki)
        Rb *= (A0 * np.exp(-(abs(xi - mu0) / sigma0)**alpha0)
               - A1 * np.exp(-(abs(xi - mu1) / sigma1)**2)
               + (1.0 + (ki / ks)**alpha2)**beta2)
    return Rb


# ---------------------------------------------------------------------------
# Vectorized helpers (one redshift, many triangles)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def bispec_array(k1_arr, k2_arr, k3_arr, z, D1, r_sigma, n_eff,
                 sigma8, om, ow, w, h, omc, omb, ns, norm,
                 k_data, pk_data, use_data):
    n = k1_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = bispec_precomp(k1_arr[i], k2_arr[i], k3_arr[i], z,
                                D1, r_sigma, n_eff,
                                sigma8, om, ow, w, h, omc, omb, ns, norm,
                                k_data, pk_data, use_data)
    return out


@njit(cache=True, parallel=True)
def bispec_tree_array(k1_arr, k2_arr, k3_arr, D1,
                      h, om, omc, omb, ns, norm,
                      k_data, pk_data, use_data):
    n = k1_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = bispec_tree_precomp(k1_arr[i], k2_arr[i], k3_arr[i], D1,
                                     h, om, omc, omb, ns, norm,
                                     k_data, pk_data, use_data)
    return out


@njit(cache=True, parallel=True)
def baryon_ratio_array(k1_arr, k2_arr, k3_arr, z):
    n = k1_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = baryon_ratio(k1_arr[i], k2_arr[i], k3_arr[i], z)
    return out


@njit(cache=True, parallel=True)
def linear_pk_array(k_arr, h, om, omc, omb, ns, norm, k_data, pk_data, use_data):
    n = k_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = linear_pk(k_arr[i], h, om, omc, omb, ns, norm,
                           k_data, pk_data, use_data)
    return out
