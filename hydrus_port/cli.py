"""Unified `hydrus` CLI — dispatches to 1D / 2D / 3D solvers.

Usage
-----

    hydrus 1d <input_dir> [-o OUT]              HYDRUS-1D
    hydrus 2d <input_dir> [-o OUT] [...]        SWMS_2D
    hydrus 3d [<input_dir>] [-o OUT]            Richards 3D (scikit-fem)

Each subcommand defaults `-o` to `<input_dir>/out`. For 3D the input
positional is optional and defaults to running the built-in synthetic
box demo from tests/validate_richards3d.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _default_out(input_dir: Path | None) -> Path:
    if input_dir is None:
        return Path.cwd() / "hydrus_out"
    return input_dir / "out"


def _run_1d(args: argparse.Namespace) -> int:
    from hydrus1d.hydrus import run_simulation
    out = args.output_dir or _default_out(args.input_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_simulation(input_dir=str(args.input_dir), output_dir=str(out))
    return 0


def _run_2d(args: argparse.Namespace) -> int:
    from swms2d.swms2d import SWMS2DSimulation
    out = args.output_dir or _default_out(args.input_dir)
    sim = SWMS2DSimulation(
        input_dir=args.input_dir,
        output_dir=out,
        use_anderson=args.anderson,
        anderson_m=args.anderson_m,
        use_banded=args.banded,
        write_vtk=args.vtk,
    )
    sim.run(verbose=not args.quiet)
    return 0


def _run_3d(args: argparse.Namespace) -> int:
    # The current 3D entry point is the validation demo. When a real
    # mesh-driven scenario format is defined we can dispatch on that.
    if args.input_dir is None:
        from tests.validate_richards3d import main as r3d_demo
        return r3d_demo()
    raise SystemExit(
        "3D mesh-driven inputs are not wired in yet. For now run "
        "`hydrus 3d` (no args) to execute the synthetic box demo."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hydrus",
        description="HYDRUS-port unified entry: 1D / 2D / 3D Richards.",
    )
    sub = p.add_subparsers(dest="kind", required=True, metavar="{1d,2d,3d}")

    # ----- 1d ---------------------------------------------------------
    p1d = sub.add_parser(
        "1d",
        help="HYDRUS-1D vertical column (Selector.in + Profile.dat)",
    )
    p1d.add_argument("input_dir", type=Path,
                     help="Directory containing Selector.in and Profile.dat")
    p1d.add_argument("-o", "--output-dir", type=Path, default=None)
    p1d.set_defaults(func=_run_1d)

    # ----- 2d ---------------------------------------------------------
    p2d = sub.add_parser(
        "2d",
        help="SWMS_2D 2D Richards / solute transport",
    )
    p2d.add_argument("input_dir", type=Path,
                     help="Directory containing SELECTOR.IN and GRID.IN")
    p2d.add_argument("-o", "--output-dir", type=Path, default=None)
    p2d.add_argument("--quiet", action="store_true",
                     help="Suppress per-step progress output")
    p2d.add_argument("--vtk", action="store_true",
                     help="Also write VTK time series next to text outputs")
    p2d.add_argument("--anderson", action="store_true",
                     help="Anderson(m) Picard acceleration (experimental)")
    p2d.add_argument("--anderson-m", type=int, default=3,
                     help="Anderson buffer depth (default 3)")
    p2d.add_argument("--banded", action="store_true",
                     help="Fortran banded-Gauss solver (else SciPy spsolve)")
    p2d.set_defaults(func=_run_2d)

    # ----- 3d ---------------------------------------------------------
    p3d = sub.add_parser(
        "3d",
        help="3D Richards (scikit-fem). With no args, runs box demo.",
    )
    p3d.add_argument("input_dir", type=Path, nargs="?", default=None,
                     help="Reserved for future mesh-driven scenarios")
    p3d.add_argument("-o", "--output-dir", type=Path, default=None)
    p3d.set_defaults(func=_run_3d)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
