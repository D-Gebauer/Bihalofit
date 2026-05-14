"""Command-line interface for BiHalofit.

The CLI mirrors the workflow demonstrated in the original C++ ``main.cpp``:
choose a cosmology, optionally load a tabulated linear P(k), then evaluate
the bispectrum (and growth factor / r_sigma) at the requested triangles and
redshifts.

Examples
--------
Single triangle, default Planck 2015 cosmology, bundled P(k) table::

    bihalofit --pk planck2015 -k 1.0 1.5 2.0 -z 0.4

Scan a series of equilateral triangles to stdout (CSV-like)::

    bihalofit -k 0.1 0.1 0.1 -k 1.0 1.0 1.0 -k 5.0 5.0 5.0 -z 0.0 -z 1.0

Custom flat wCDM cosmology with Eisenstein-Hu P(k)::

    bihalofit --omb 0.047 --omc 0.286 --h 0.7 --sigma8 0.82 \\
              --ns 0.96 --w -1 -k 1 1.5 2 -z 0.4
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

from .core import BiHalofit, DEFAULTS, planck2015_pk_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bihalofit",
        description=(
            "Compute the non-linear matter bispectrum from BiHalofit "
            "(Takahashi+ 2019, arXiv:1911.07886)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cosmo = p.add_argument_group("cosmology (flat wCDM)")
    cosmo.add_argument("--omb", type=float, default=DEFAULTS["omb"],
                       help="Omega_baryon")
    cosmo.add_argument("--omc", type=float, default=DEFAULTS["omc"],
                       help="Omega_cdm")
    cosmo.add_argument("--h", type=float, default=DEFAULTS["h"],
                       help="dimensionless Hubble parameter")
    cosmo.add_argument("--sigma8", type=float, default=DEFAULTS["sigma8"],
                       help="linear sigma8")
    cosmo.add_argument("--ns", type=float, default=DEFAULTS["ns"],
                       help="spectral index n_s")
    cosmo.add_argument("--w", type=float, default=DEFAULTS["w"],
                       help="dark-energy equation of state")

    pk = p.add_argument_group("linear power spectrum")
    pk.add_argument(
        "--pk",
        default=None,
        help=(
            "path to a two-column (k[h/Mpc], P[(Mpc/h)^3]) ASCII table. "
            "Pass 'planck2015' to use the bundled Planck 2015 table; "
            "omit entirely to use the Eisenstein & Hu (1999) fitting formula."
        ),
    )

    p.add_argument(
        "-k", "--triangle",
        nargs=3, type=float, metavar=("K1", "K2", "K3"),
        action="append", required=True,
        help="triangle (k1, k2, k3) in h/Mpc; may be repeated.",
    )
    p.add_argument(
        "-z", "--redshift",
        type=float, action="append", required=True,
        help="redshift; may be repeated to evaluate at multiple z.",
    )
    p.add_argument(
        "--no-baryons", action="store_true",
        help="omit the baryon-ratio column.",
    )
    p.add_argument(
        "--header", action="store_true",
        help="print a header row before the data rows.",
    )
    p.add_argument(
        "--sep", default="\t",
        help="output column separator.",
    )
    return p


def _resolve_pk(path: str | None) -> str | None:
    if path is None:
        return None
    if path.lower() == "planck2015":
        return str(planck2015_pk_path())
    return path


def _format_row(values: List[float], sep: str) -> str:
    return sep.join(f"{v:.6e}" for v in values)


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bh = BiHalofit(
        omb=args.omb, omc=args.omc, h=args.h,
        sigma8=args.sigma8, ns=args.ns, w=args.w,
    )
    pk_path = _resolve_pk(args.pk)
    if pk_path is not None:
        bh.load_pk_data(pk_path)

    triangles: List[Tuple[float, float, float]] = [tuple(t) for t in args.triangle]
    zs: List[float] = list(args.redshift)

    columns = ["z", "k1", "k2", "k3", "D1", "r_sigma", "bispec", "bispec_tree"]
    if not args.no_baryons:
        columns.append("baryon_ratio")
    if args.header:
        print(args.sep.join(columns))

    for z in zs:
        D1 = bh.D1(z)
        rs = bh.r_sigma(z)
        for (k1, k2, k3) in triangles:
            if k1 > k2 + k3 or k2 > k1 + k3 or k3 > k1 + k2:
                print(
                    f"# warning: (k1,k2,k3)=({k1},{k2},{k3}) violates the "
                    "triangle inequality; output will be unphysical.",
                    file=sys.stderr,
                )
            row = [z, k1, k2, k3, D1, rs,
                   bh.bispec(k1, k2, k3, z),
                   bh.bispec_tree(k1, k2, k3, z)]
            if not args.no_baryons:
                row.append(bh.baryon_ratio(k1, k2, k3, z))
            print(_format_row(row, args.sep))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
