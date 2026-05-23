"""Round-trip test: parse SELECTOR.IN → write → parse → assert equal.

This is the gate the GUI parameter editor relies on:
  EX1 loaded → some field edited → file regenerated → re-loaded → same.
We require *semantic* equality, not byte-equal whitespace.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swms2d.input import parse_selector, write_selector
import numpy as np


def _eq_floats(a, b, tol=1e-9):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(a, b, atol=tol, rtol=tol)
    return abs(float(a) - float(b)) < tol


def _cfg_eq(a, b) -> list[str]:
    diffs: list[str] = []
    for k in vars(a):
        va, vb = getattr(a, k, None), getattr(b, k, None)
        if isinstance(va, float) or isinstance(vb, float):
            if not _eq_floats(va, vb):
                diffs.append(f"cfg.{k}: {va} != {vb}")
        elif va != vb:
            diffs.append(f"cfg.{k}: {va!r} != {vb!r}")
    return diffs


def _materials_eq(ma, mb) -> list[str]:
    if len(ma) != len(mb):
        return [f"materials length {len(ma)} != {len(mb)}"]
    diffs: list[str] = []
    for i, (a, b) in enumerate(zip(ma, mb)):
        for k in vars(a):
            va = getattr(a, k); vb = getattr(b, k)
            if isinstance(va, float):
                if not _eq_floats(va, vb):
                    diffs.append(f"mat[{i}].{k}: {va} != {vb}")
            elif va != vb:
                diffs.append(f"mat[{i}].{k}: {va} != {vb}")
    return diffs


def _time_eq(ta, tb) -> list[str]:
    diffs: list[str] = []
    for k in vars(ta):
        va, vb = getattr(ta, k), getattr(tb, k)
        if isinstance(va, float):
            if not _eq_floats(va, vb):
                diffs.append(f"time.{k}: {va} != {vb}")
        elif isinstance(va, np.ndarray):
            if not _eq_floats(va, vb):
                diffs.append(f"time.{k}: array differs")
        elif va != vb:
            diffs.append(f"time.{k}: {va} != {vb}")
    return diffs


def _extras_eq(ea, eb) -> list[str]:
    diffs: list[str] = []
    keys = set(ea) | set(eb)
    skip = {"chem_remaining_lines"}
    for k in keys - skip:
        va, vb = ea.get(k), eb.get(k)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not _eq_floats(np.asarray(va), np.asarray(vb)):
                diffs.append(f"extras.{k}: array differs")
        elif isinstance(va, float) or isinstance(vb, float):
            if not _eq_floats(va, vb):
                diffs.append(f"extras.{k}: {va} != {vb}")
        elif isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                diffs.append(f"extras.{k}: length {len(va)} != {len(vb)}")
            for i, (x, y) in enumerate(zip(va, vb)):
                if x != y:
                    diffs.append(f"extras.{k}[{i}]: {x} != {y}")
        elif va != vb:
            diffs.append(f"extras.{k}: {va!r} != {vb!r}")
    return diffs


def round_trip(in_dir: Path) -> tuple[bool, list[str]]:
    sel_in = in_dir / "SELECTOR.IN"
    if not sel_in.exists():
        # Fallback case-insensitive
        sel_in = next(p for p in in_dir.iterdir()
                      if p.name.lower() == "selector.in")
    cfg1, mats1, time1, extras1 = parse_selector(sel_in)
    tmp = in_dir / "_SELECTOR_RT.IN"
    try:
        write_selector(cfg1, mats1, time1, extras1, tmp)
        cfg2, mats2, time2, extras2 = parse_selector(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    diffs: list[str] = []
    diffs += _cfg_eq(cfg1, cfg2)
    diffs += _materials_eq(mats1, mats2)
    diffs += _time_eq(time1, time2)
    diffs += _extras_eq(extras1, extras2)
    return (not diffs), diffs


def main():
    fixtures = ROOT / "tests" / "fixtures"
    overall = True
    for name in ("EX1", "EX2", "EX3", "EX4"):
        in_dir = fixtures / name / "inputs"
        if not in_dir.exists():
            print(f"SKIP {name}: no fixture")
            continue
        try:
            ok, diffs = round_trip(in_dir)
        except Exception as e:
            ok, diffs = False, [f"exception: {e!r}"]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            overall = False
            for d in diffs[:20]:
                print(f"  {d}")
            if len(diffs) > 20:
                print(f"  ... ({len(diffs) - 20} more)")
    print("\nOVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
