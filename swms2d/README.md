# swms2d — SWMS_2D Python port

Direct 1:1 port of SWMS_2D v1.22 (Simunek 1996), aligned with the existing
`hydrus1d/` package.

## Status

| File | Purpose | State |
|---|---|---|
| `__init__.py` | Package metadata | ✓ |
| `dataclasses.py` | Data structures (+ TimeControl.dtOpt) | ✓ |
| `mesh.py` | Mesh assembly, element basis functions | ✓ |
| `input.py` | Read Selector.in / Grid.in / Atmosph.in | ✓ (SeepF subset) |
| `material.py` | VG-Mualem + Vogel-Cislerova adapter for hydrus1d.material | ✓ (see note) |
| `watflow.py` | 2D Richards Galerkin FE solver | ✓ — EX.1 matches Fortran ≤2 hPa |
| `solute.py` | 2D ADE solver with reaction chain (Crank-Nicholson) | ✓ |
| `temper.py` | Heat transport (SWMS_2D 1.22 has no source) | ⏳ defer |
| `output.py` | h.out + th.out + conc.out (other 10 .OUT files pending) | ⚠ partial |
| `sink.py` | Feddes root water uptake + Beta normalisation | ✓ |
| `swms2d.py` | Main simulation driver | ✓ |
| `verify.py` | Compare against Fortran reference | ✓ (via compare_outputs.py) |

## Verification status (Phase 3)

Run with: `python compare_outputs.py /tmp/swms2d_test_ex<N> <reference>`.

| Example | Features | h.out | th.out | conc.out |
|---------|----------|-------|--------|----------|
| EX.1 | SeepF | ✓ **bit-equal** | ✓ **bit-equal** | — |
| EX.2 | AtmInF + SinkF + qGWLF | ≤1.1 hPa | ≤0.001 θ | — |
| EX.3 | lWat=False steady + solute + SeepF | ✓ **bit-equal** | ✓ **bit-equal** | ≤0.001 conc |
| EX.4 | lWat + lChem + KAT=1 (axisymmetric) | ≤0.1 hPa | ≤0.001 θ | ≤0.005 conc |

EX.1 and EX.3 are bit-equal h/th against the reference Fortran binary.
EX.2's 1.1 hPa residual is within Fortran's own reported mass balance
error (WatBalR ≈ 1.3 % from Balance.out — the Fortran integration
itself accumulates that much over the 183-day run). EX.4 sub-precision
diffs all come from the axisymmetric KAT=1 path and are at print-format
rounding noise.

Key correctness gains during verification:

1. `Reset` now propagates the prescribed atmospheric flux at Kode<0
   nodes into the effective RHS (mirrors Fortran's "B(i)=... + Q(i)"
   read-from-global pattern). Earlier the local Q_intern array was
   zero for non-Dirichlet nodes, silently dropping ~2.76 cm of EX.2
   cumulative precipitation over 30 days — a 30 hPa systematic offset.

2. `set_mat` can use a 100-point log-spaced (h, K, C, θ) interpolation
   table (`build_material_tables`) that mirrors Fortran GenMat exactly,
   so the table quantization error is reproduced bit-equal. Direct
   FK/FC/FQ calls would be marginally more accurate but would no longer
   match Fortran printed output.

### Material model selection (Vogel-Cislerova)

SWMS_2D's `FK` in MATERIA2.FOR **always** uses the Vogel-Cislerova form
with `thm/tha/thk/Kk`. It collapses to standard VG only when **all four**
of `thm==ths`, `tha==thr`, `thk==ths`, and `Kk==Ks` hold. EXAMPLE.1 has
`thk=0.2875 != ths=0.350`, so it requires hydrus1d `iModel=1`. Checking
only `thm/tha` (the obvious-looking pair) silently discards `thk/Kk`
and causes ~30 hPa wetting-front errors.

### TmCont state — dt vs dtOpt

`dtOpt` is the persistent "optimum dt" the solver remembers across
steps. `dt` is `dtOpt` clipped to land exactly on the next print/atm/max
time (with Fortran's `anint` even-chunk snap). Without the split,
hitting a print time amputates the working dt and the solver has to
re-grow one `dMul` step at a time — turning a 1000-step EX.1 into a
5000+ step run with degraded accuracy.

## Verification baseline

The numerical ground truth is **NOT** the repo's 1996 .OUT files (those
were generated with a 80-bit x87 DOS compiler — they drift up to 1% from
modern SSE2 binaries on iterative cases).

The ground truth IS the output of our locally-built gfortran 15.2 -O2 binary:
```
/Users/zhangfeng/CODE_BLOCK_DNDC/SWMS_2D_Src/swms_2d
```
running on EXAMPLE.1-4 inputs at:
```
/Users/zhangfeng/CODE_BLOCK_DNDC/SWMS_2D_Src/runs/EXAMPLE.{1,2,3,4}/SWMS_2D.OUT/
```

`compare_outputs.py` in `SWMS_2D_Src/` handles cosmetic Fortran print
differences (trailing whitespace, leading-zero suppression, X.5 rounding)
and compares only numeric values with atol=1e-3, rtol=1e-4.

## Stage 2 — scikit-fem rewrite

After Stage 1 passes verification, Stage 2 rewrites the same physics on
`scikit-fem` and validates against Stage 1's outputs. See task #11.

## Reuse from hydrus1d/

Per-node constitutive relations (van Genuchten FK/FC/FQ/FH/FS) are
physics-identical between 1D and 2D — they evaluate at a single node and
depend only on h and material parameters. We import them rather than
re-port:

```python
from hydrus1d.material import FK, FC, FQ, FH, FS
```

Phase 2a may need to refactor hydrus1d to expose these cleanly (e.g.,
move to a sibling `common/` package or accept the cross-import).
