"""
Compare HYDRUS-1D Python port output against the Fortran golden reference.

Usage:
    python tests/compare_outputs.py <fixture_dir> [--rtol 1e-3] [--atol 1e-6]

Where <fixture_dir> contains both ``reference_out/`` (golden Fortran output)
and ``python_out/`` (Python run output). The script reports a pass/fail per
file plus per-quantity error statistics.

Comparisons focus on the physical observables, not formatting:

- ``BALANCE.OUT``  →  W-volume, h Mean, Top Flux, Bot Flux, WatBalT, WatBalR
                     at each print time.
- ``T_LEVEL.OUT``  →  time series of vTop, vBot, sum(vTop), sum(vBot),
                     hTop, hBot, Volume (interpolated to common timestamps).
- ``NOD_INF.OUT``  →  per-node Head, Moisture, K, Flux at the *last* print
                     time only (matching profile is enough for sanity).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_BALANCE_KEYS = ["W-volume", "h Mean", "Top Flux", "Bot Flux", "WatBalT", "WatBalR"]


def parse_balance(path: Path) -> Dict[float, Dict[str, float]]:
    """Return ``{t: {key: value}}`` for each time block in BALANCE.OUT."""
    txt = path.read_text(errors="replace")
    blocks: Dict[float, Dict[str, float]] = {}
    current_t = None
    current = {}
    for raw in txt.splitlines():
        m_t = re.match(r"\s*Time\s*\[T\]\s+([-\d.E+]+)", raw)
        if m_t:
            if current_t is not None:
                blocks[current_t] = current
            current_t = float(m_t.group(1))
            current = {}
            continue
        for key in _BALANCE_KEYS:
            if raw.lstrip().startswith(key):
                # First numeric token after the key is the value of interest
                rest = raw.split(key, 1)[1]
                m = re.search(r"[-+]?\d+\.\d+E?[-+]?\d*|[-+]?\d+\.\d+|[-+]?\d+", rest)
                if m:
                    current[key] = float(m.group(0))
                break
    if current_t is not None:
        blocks[current_t] = current
    return blocks


def parse_t_level(path: Path) -> np.ndarray:
    """Return an (N, 6) array of [time, vTop, vBot, hTop, hBot, sum_vTop]."""
    rows = []
    started = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("end"):
            continue
        toks = line.split()
        if not started:
            # Look for a row whose first token parses as a float >= 0 — that's
            # the first data row (header lines start with letters).
            try:
                float(toks[0])
                if not re.match(r"^[-+]?\d", toks[0]):
                    continue
                started = True
            except (ValueError, IndexError):
                continue
        try:
            t = float(toks[0])
        except ValueError:
            continue
        if len(toks) < 12:
            continue
        try:
            # Fortran TLInf column order:
            #   t, rTop, rRoot, vTop, vRoot, vBot, sum_rTop, sum_rRoot,
            #   sum_vTop, sum_vRoot, sum_vBot, hTop, hRoot, hBot, ...
            vTop = float(toks[3]); vBot = float(toks[5])
            sumVtop = float(toks[8]); sumVbot = float(toks[10])
            hTop = float(toks[11]); hBot = float(toks[13])
            rows.append([t, vTop, vBot, hTop, hBot, sumVtop])
        except (ValueError, IndexError):
            continue
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6))


def parse_nod_inf_last(path: Path) -> Tuple[float, np.ndarray]:
    """Return (last_time, array(N, 4)) with columns [Depth, Head, Moisture, K]."""
    txt = path.read_text(errors="replace")
    blocks = re.split(r"\n\s*Time:\s+", txt)
    if len(blocks) < 2:
        return 0.0, np.zeros((0, 4))
    last = blocks[-1]
    header_line, rest = last.split("\n", 1)
    try:
        t = float(header_line.strip().split()[0])
    except ValueError:
        return 0.0, np.zeros((0, 4))
    rows = []
    for raw in rest.splitlines():
        line = raw.strip()
        if not line or line.startswith("end") or line.startswith("Node"):
            continue
        if line[0].isalpha():
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            depth = float(toks[1])
            head = float(toks[2])
            theta = float(toks[3])
            K = float(toks[4])
            rows.append([depth, head, theta, K])
        except ValueError:
            continue
    return t, np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def relative_err(a: float, b: float) -> float:
    # When both values are tiny in absolute terms (≤ atol scaled by 1e3)
    # the relative measure is meaningless — return 0.
    if max(abs(a), abs(b)) < 1e-6:
        return 0.0
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def compare_balance(ref: Path, py: Path, rtol: float, atol: float) -> bool:
    print(f"\n--- BALANCE.OUT ---")
    if not py.exists():
        print(f"  MISSING: {py}")
        return False
    R = parse_balance(ref)
    P = parse_balance(py)
    common = sorted(set(R.keys()) & set(P.keys()))
    if not common:
        print("  no common time points")
        return False
    ok = True
    for t in common:
        for k in _BALANCE_KEYS:
            r = R[t].get(k); p = P[t].get(k)
            if r is None or p is None:
                continue
            rel = relative_err(r, p)
            abs_diff = abs(r - p)
            tag = "OK  " if (abs_diff <= atol or rel <= rtol) else "FAIL"
            if tag.startswith("FAIL"):
                ok = False
            print(f"  t={t:8.3f}  {k:<12}  ref={r:13.5g}  py={p:13.5g}  rel={rel:8.2e}  {tag}")
    return ok


def compare_tlevel(ref: Path, py: Path, rtol: float, atol: float) -> bool:
    print(f"\n--- T_LEVEL.OUT ---")
    if not py.exists():
        print(f"  MISSING: {py}")
        return False
    R = parse_t_level(ref)
    P = parse_t_level(py)
    if R.size == 0 or P.size == 0:
        print(f"  no rows parsed (ref={R.shape}, py={P.shape})")
        return False
    # Interpolate Python series to reference timestamps.
    t_ref = R[:, 0]
    ok = True
    cols = {1: "vTop", 2: "vBot", 3: "hTop", 4: "hBot", 5: "sum_vTop"}
    for c, name in cols.items():
        p_interp = np.interp(t_ref, P[:, 0], P[:, c])
        denom = np.maximum(np.abs(R[:, c]), np.abs(p_interp))
        denom = np.where(denom < 1e-30, 1e-30, denom)
        rel = np.abs(R[:, c] - p_interp) / denom
        worst = int(np.argmax(rel))
        tag = "OK  " if (rel[worst] <= rtol or
                         np.abs(R[worst, c] - p_interp[worst]) <= atol) else "FAIL"
        if tag.startswith("FAIL"):
            ok = False
        print(f"  {name:10s}  worst@t={t_ref[worst]:8.3f}  "
              f"ref={R[worst, c]:13.5g}  py={p_interp[worst]:13.5g}  "
              f"rel={rel[worst]:8.2e}  {tag}")
    return ok


def compare_nod_inf(ref: Path, py: Path, rtol: float, atol: float) -> bool:
    print(f"\n--- NOD_INF.OUT (last print) ---")
    if not py.exists():
        print(f"  MISSING: {py}")
        return False
    tR, R = parse_nod_inf_last(ref)
    tP, P = parse_nod_inf_last(py)
    if R.size == 0 or P.size == 0:
        print(f"  no nodes parsed (ref={R.shape}, py={P.shape})")
        return False
    print(f"  ref t={tR:.3f}, py t={tP:.3f}, N_ref={len(R)}, N_py={len(P)}")
    n = min(len(R), len(P))
    R = R[:n]; P = P[:n]
    ok = True
    for j, name in enumerate(("Depth", "Head", "Moisture", "K")):
        if j == 0:
            continue
        denom = np.maximum(np.abs(R[:, j]), np.abs(P[:, j]))
        denom = np.where(denom < 1e-30, 1e-30, denom)
        rel = np.abs(R[:, j] - P[:, j]) / denom
        rms = np.sqrt(np.mean(rel ** 2))
        worst = int(np.argmax(rel))
        tag = "OK  " if rms <= rtol else "FAIL"
        if tag.startswith("FAIL"):
            ok = False
        print(f"  {name:10s}  rms_rel={rms:8.2e}  worst rel={rel[worst]:8.2e} "
              f"at depth={R[worst, 0]:9.3f}  {tag}")
    return ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir", type=Path,
                    help="Path to fixture dir containing reference_out/ + python_out/")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    ref_dir = args.fixture_dir / "reference_out"
    py_dir = args.fixture_dir / "python_out"
    if not ref_dir.is_dir():
        print(f"missing reference_out: {ref_dir}", file=sys.stderr)
        return 2
    if not py_dir.is_dir():
        print(f"missing python_out: {py_dir}", file=sys.stderr)
        return 2

    pass_bal = compare_balance(ref_dir / "BALANCE.OUT",
                               py_dir / "BALANCE.OUT", args.rtol, args.atol)
    pass_t = compare_tlevel(ref_dir / "T_LEVEL.OUT",
                            py_dir / "T_LEVEL.OUT", args.rtol, args.atol)
    pass_nod = compare_nod_inf(ref_dir / "NOD_INF.OUT",
                               py_dir / "NOD_INF.OUT", args.rtol, args.atol)
    print()
    print(f"BALANCE: {'PASS' if pass_bal else 'FAIL'}")
    print(f"T_LEVEL: {'PASS' if pass_t  else 'FAIL'}")
    print(f"NOD_INF: {'PASS' if pass_nod else 'FAIL'}")
    return 0 if (pass_bal and pass_t and pass_nod) else 1


if __name__ == "__main__":
    sys.exit(main())
