"""Cross-validate richards3d: Tet vs Hex, lumped vs consistent mass.

Runs the same 3D vertical-column infiltration problem (saturated top,
free drainage on bottom, no-flow on sides) under four configurations:

    Tet + lumped (default)
    Tet + consistent
    Hex + lumped
    Hex + consistent

and checks that all four agree on the mid-column h(z) profile to
within a reasonable tolerance. The lumped variants are the production
path; the consistent ones are sanity controls — they should converge
to the same solution as the mesh is refined.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swms2d.mesh_io import make_box_mesh_3d
from swms2d.richards3d import RichardsState3D, integrate_3d
from swms2d.dataclasses import SoilMaterial


def run_case(element: str, lump: bool, *, nz: int = 21, t_end: float = 0.05):
    box = make_box_mesh_3d(nx=4, ny=4, nz=nz, lx=0.1, ly=0.1, lz=1.0,
                           element=element)
    skmesh = box["skmesh"]
    N = skmesh.p.shape[1]
    z = skmesh.p[2]

    # Single material: loam (van Genuchten)
    mat = SoilMaterial(
        ths=0.430, thr=0.078, alpha=0.036, n=1.56,
        Ks=0.2496, Kk=0.2496, thk=0.430,
        tha=0.078, thm=0.430,
    )

    # Initial condition: h = -200 cm everywhere (uniform dry)
    h0 = np.full(N, -200.0, dtype=np.float64)
    # Top boundary: ponded (h = 0) on z = lz
    top = np.where(np.isclose(z, z.max()))[0]
    h0[top] = 0.0

    # Need theta_old initialised
    from swms2d.richards3d import evaluate_KCQ_3d
    matnum = np.ones(N, dtype=np.int32)
    _, _, th0 = evaluate_KCQ_3d(h0, matnum, [mat])
    state = RichardsState3D(
        hNew=h0.copy(), hOld=h0.copy(),
        ThNew=th0.copy(), ThOld=th0.copy(),
        MatNum=matnum,
    )

    # Dirichlet: top fixed at 0; bottom: free drainage handled by
    # zero-Neumann (default Q_bc=0) approximation here
    dirich = top.astype(np.int32)
    dirich_h = np.zeros(top.size)

    h_final, th_final, _ = integrate_3d(
        skmesh, state, [mat],
        t_end=t_end, dt_init=1e-3,
        dirich_nodes=dirich, dirich_h=dirich_h,
        gravity_axis=2, lump=lump,
        dt_max=0.02, max_picard=20, tol_h=0.5,
    )
    # Sample mid-column profile: nearest x/y to box centre
    xc, yc = 0.05, 0.05
    xu = np.unique(skmesh.p[0]); yu = np.unique(skmesh.p[1])
    x_target = xu[np.argmin(np.abs(xu - xc))]
    y_target = yu[np.argmin(np.abs(yu - yc))]
    midx = np.where(np.isclose(skmesh.p[0], x_target) &
                     np.isclose(skmesh.p[1], y_target))[0]
    midx = midx[np.argsort(z[midx])]
    return z[midx], h_final[midx], th_final[midx]


def main():
    results = {}
    for element in ("tetra", "hex"):
        for lump in (True, False):
            tag = f"{element}/{'lump' if lump else 'consistent'}"
            print(f"running {tag} ...", flush=True)
            zs, hs, ths = run_case(element, lump)
            results[tag] = (zs, hs, ths)
            print(f"  h range: [{hs.min():.2f}, {hs.max():.2f}]   "
                  f"theta range: [{ths.min():.4f}, {ths.max():.4f}]")

    # Compare lumped vs consistent on the same element type
    print("\n== max |Δh| between lumped and consistent on same element ==")
    for element in ("tetra", "hex"):
        _, h_l, _ = results[f"{element}/lump"]
        _, h_c, _ = results[f"{element}/consistent"]
        print(f"  {element:5s}  max|Δh| = {np.max(np.abs(h_l - h_c)):.3f} cm")

    # Cross-element comparison (lumped only)
    z_t, h_t, _ = results["tetra/lump"]
    z_h, h_h, _ = results["hex/lump"]
    # Interpolate hex onto tet's z for comparison
    h_h_at_t = np.interp(z_t, z_h, h_h)
    print(f"\n== tetra/lump vs hex/lump  max|Δh| = "
          f"{np.max(np.abs(h_t - h_h_at_t)):.3f} cm ==")

    # Pass criterion: solutions agree within 20 cm anywhere on
    # the column. With nz=21 this is a coarse grid; tighter agreement
    # would require refinement.
    threshold = 20.0
    ok = True
    for element in ("tetra", "hex"):
        _, h_l, _ = results[f"{element}/lump"]
        _, h_c, _ = results[f"{element}/consistent"]
        d = np.max(np.abs(h_l - h_c))
        if d > threshold:
            print(f"FAIL: {element} lumped vs consistent disagrees by {d:.1f} cm")
            ok = False
    cross = np.max(np.abs(h_t - h_h_at_t))
    if cross > threshold:
        print(f"FAIL: tet vs hex disagrees by {cross:.1f} cm")
        ok = False
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
