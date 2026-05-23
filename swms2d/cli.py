"""Command-line entry point for the swms2d package."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .swms2d import SWMS2DSimulation


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="swms2d",
        description="SWMS_2D Python port — 2D Richards / solute transport",
    )
    p.add_argument("input_dir", type=Path,
                   help="Directory containing SWMS_2D.IN inputs")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Output directory (default: <input_dir>/out)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-step progress output")
    p.add_argument("--vtk", action="store_true",
                   help="Also write VTK time series next to text outputs")
    # Opt-in experimental flags (parity with SWMS2DSimulation kwargs)
    p.add_argument("--anderson", action="store_true",
                   help="Enable Anderson(m) Picard acceleration")
    p.add_argument("--anderson-m", type=int, default=3,
                   help="Anderson buffer depth (default 3)")
    p.add_argument("--banded", action="store_true",
                   help="Use Fortran banded-Gauss solver (else SciPy spsolve)")
    args = p.parse_args(argv)

    output_dir = args.output_dir or (args.input_dir / "out")
    sim = SWMS2DSimulation(
        input_dir=args.input_dir,
        output_dir=output_dir,
        use_anderson=args.anderson,
        anderson_m=args.anderson_m,
        use_banded=args.banded,
        write_vtk=args.vtk,
    )
    sim.run(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
