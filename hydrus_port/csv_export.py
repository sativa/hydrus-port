"""Convert HYDRUS-1D / SWMS_2D / Richards3D outputs to long-format CSV.

The simulators write Fortran-style ASCII .OUT files that haven't aged
well — multi-block layouts, fixed-column widths, "Time:" snapshot
headers buried in numeric rows. This module gives you tidy CSVs you
can `pandas.read_csv` and pivot however you want.

Public API
----------

    convert_output_dir(out_dir) -> list[Path]
        Detects every .out/.OUT/.dat we know how to convert and emits
        the matching .csv next to it. Returns paths of written CSVs.

    --- per-file converters (use directly when you only want one) ---
    convert_t_level(path)         HYDRUS-1D time-series
    convert_balance_h1d(path)     HYDRUS-1D mass balance
    convert_nod_inf(path)         HYDRUS-1D per-node × per-time
    convert_balance_swms(path)    SWMS_2D mass-balance blocks
    convert_run_inf_swms(path)    SWMS_2D time-step inf
    convert_swms_field(path, …)   SWMS_2D h.out / th.out / Q.out
    convert_swms_velocity(vx_path, vz_path)
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------------
# Tiny helpers
# ----------------------------------------------------------------------

def _is_numeric_row(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("*"):
        return False
    first = s.split()[0]
    try:
        float(first); return True
    except ValueError:
        try:
            int(first); return True
        except ValueError:
            return False


def _looks_like_units(toks: list[str]) -> bool:
    return all(t.startswith("[") and t.endswith("]") for t in toks)


def _write_csv(path: Path, header: list[str], rows: Iterable[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------
# HYDRUS-1D: T_LEVEL.OUT (22 columns of time series — already tidy)
# ----------------------------------------------------------------------

H1D_TLEVEL_COLS = [
    "Time", "rTop", "rRoot", "vTop", "vRoot", "vBot",
    "sum_rTop", "sum_rRoot", "sum_vTop", "sum_vRoot", "sum_vBot",
    "hTop", "hRoot", "hBot", "RunOff", "sum_RunOff",
    "Volume", "sum_Infil", "sum_Evap", "TLevel", "Cum_WTrans", "SnowLayer",
]


def convert_t_level(path: Path) -> Path:
    out = path.with_suffix(".csv")
    rows = []
    for line in path.read_text().splitlines():
        if _is_numeric_row(line):
            vals = line.split()
            if len(vals) >= 5:
                rows.append([float(v) if "." in v or "e" in v.lower() else
                             (int(v) if v.lstrip("-").isdigit() else float(v))
                             for v in vals])
    width = max((len(r) for r in rows), default=0)
    header = H1D_TLEVEL_COLS[:width] if width else H1D_TLEVEL_COLS
    while len(header) < width:
        header.append(f"col{len(header) + 1}")
    _write_csv(out, header, rows)
    return out


# ----------------------------------------------------------------------
# HYDRUS-1D: BALANCE.OUT (multi-time-window blocks)
# ----------------------------------------------------------------------

def convert_balance_h1d(path: Path) -> Path:
    """Long format: one row per (time_window, quantity)."""
    out = path.with_suffix(".csv")
    text = path.read_text()
    rows: list[list] = []
    current_t: float | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        m = re.search(r"Time\s*\[T\]\s+([0-9.eE+\-]+)", line)
        if m:
            current_t = float(m.group(1))
            continue
        # Quantity: name value [value …]
        # Examples:
        #   "WatBalT [V]            -7.98e-01"
        toks = s.split()
        if len(toks) >= 2 and current_t is not None:
            # Try to find a numeric tail
            for split in range(len(toks)):
                tail = toks[split:]
                if all(re.match(r"^[-+]?\d", t) for t in tail):
                    name = " ".join(toks[:split])
                    for col, val in enumerate(tail):
                        try:
                            rows.append([current_t, name, col, float(val)])
                        except ValueError:
                            pass
                    break
    _write_csv(out, ["time", "quantity", "col", "value"], rows)
    return out


# ----------------------------------------------------------------------
# HYDRUS-1D: NOD_INF.OUT (multiple per-node snapshots over time)
# ----------------------------------------------------------------------

NOD_INF_COLS = [
    "Node", "Depth", "Head", "Moisture", "K", "C",
    "Flux", "Sink", "Kappa", "vOverKsTop", "Temp",
]


def convert_nod_inf(path: Path) -> Path:
    """Long format: one row per (time, node)."""
    out = path.with_suffix(".csv")
    rows: list[list] = []
    current_t: float | None = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Time:"):
            try:
                current_t = float(s.split(":", 1)[1].strip())
            except ValueError:
                current_t = None
            continue
        if current_t is None:
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            int(toks[0])
        except ValueError:
            continue
        # Take up to 11 numeric tokens
        vals = []
        for t in toks[:11]:
            try:
                vals.append(float(t))
            except ValueError:
                vals.append(None)
        if len(vals) >= 3:
            # Pad to NOD_INF_COLS length
            while len(vals) < len(NOD_INF_COLS):
                vals.append(None)
            rows.append([current_t, *vals])
    _write_csv(out, ["time", *NOD_INF_COLS], rows)
    return out


# ----------------------------------------------------------------------
# SWMS_2D: Balance.out (multi-time mass-balance blocks)
# ----------------------------------------------------------------------

def convert_balance_swms(path: Path) -> Path:
    """Long format: one row per (time, quantity, region).

    SWMS_2D Balance.out layout:
        Time [T]    Total   Sub-region 1   Sub-region 2 ...
        <t>                    1            2  ...    (time header row)
        Quantity [unit]   total_val   region_1_val   region_2_val ...
        ... more quantity rows for this time ...

    The time is the first numeric row right after a "Time [T]" header line.
    """
    out = path.with_suffix(".csv")
    rows: list[list] = []
    current_t: float | None = None
    expect_time_row = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        # Comment / metadata lines we discard
        if s.startswith("Program ") or "boundary conditions" in s or \
           "plane flow" in s or s.startswith("Units:") or s.startswith("Example") \
           or s.startswith("**"):
            continue
        if s.startswith("Time [T]"):
            expect_time_row = True
            continue
        # First line after "Time [T]" header: it's the time + region indices
        if expect_time_row:
            toks = s.split()
            try:
                current_t = float(toks[0])
                expect_time_row = False
                continue
            except (ValueError, IndexError):
                expect_time_row = False
                # fall through and try as quantity row
        if current_t is None:
            continue
        # Quantity row: "Name [unit]   total   region_1 ..."
        toks = s.split()
        # Find the split between name (text) and numeric tail
        for split in range(1, len(toks) + 1):
            tail = toks[split:]
            if not tail:
                break
            if all(re.match(r"^[-+]?[0-9.]", t) for t in tail):
                name = " ".join(toks[:split]).rstrip()
                # First numeric column = "Total", rest = sub-regions 1..N
                for region, val in enumerate(tail):
                    label = "total" if region == 0 else f"region_{region}"
                    try:
                        rows.append([current_t, name, label, float(val)])
                    except ValueError:
                        pass
                break
    _write_csv(out, ["time", "quantity", "region", "value"], rows)
    return out


# ----------------------------------------------------------------------
# SWMS_2D: Run_Inf.out (per-time-step solver info, tidy table)
# ----------------------------------------------------------------------

def convert_run_inf(path: Path) -> Path:
    """SWMS_2D Run_Inf.out — a fixed-column table after a 'TLevel Time
    dt Iter ItCum' header line. Pass through as flat CSV."""
    out = path.with_suffix(".csv")
    rows: list[list] = []
    header: list[str] = ["TLevel", "Time", "dt", "Iter", "ItCum"]
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        toks = s.split()
        try:
            int(toks[0])
        except (ValueError, IndexError):
            continue
        if len(toks) >= 5:
            try:
                rows.append([int(toks[0]), float(toks[1]), float(toks[2]),
                              int(toks[3]), int(toks[4])])
            except ValueError:
                pass
    _write_csv(out, header, rows)
    return out


# ----------------------------------------------------------------------
# SWMS_2D: h.out / th.out / Q.out (per-node values × time snapshots)
# Format groups two nodes per row: "n  x(n)  z(n)  h(n)  h(n+1)"
# ----------------------------------------------------------------------

def convert_swms_field(path: Path, value_name: str = "value",
                        grid_path: Path | None = None) -> Path:
    """Long format: one row per (time, node, x, z, value)."""
    out = path.with_suffix(".csv")

    # Number of nodes — read from GRID.IN if given, else infer at parse time
    num_np: int | None = None
    if grid_path and grid_path.exists():
        try:
            for ln in grid_path.read_text().splitlines():
                toks = ln.split()
                if len(toks) >= 2:
                    try:
                        num_np = int(toks[0])
                        break
                    except ValueError:
                        pass
        except Exception:
            num_np = None

    rows: list[list] = []
    current_t: float | None = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"Time\s*\*\*\*\s*([0-9.eE+\-]+)", s)
        if m:
            current_t = float(m.group(1))
            continue
        if current_t is None:
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            n = int(toks[0])
        except ValueError:
            continue
        if len(toks) < 5:
            continue
        try:
            x = float(toks[1]); z = float(toks[2])
            v1 = float(toks[3]); v2 = float(toks[4])
        except ValueError:
            continue
        rows.append([current_t, n,     x, z, v1])
        # Pair node n+1 in the same row (use same x/z — caller knows
        # the mesh's coord-grouping convention)
        if num_np is None or n < num_np:
            rows.append([current_t, n + 1, x, z, v2])
    _write_csv(out, ["time", "node", "x", "z", value_name], rows)
    return out


def convert_swms_velocity(vx_path: Path, vz_path: Path,
                            grid_path: Path | None = None) -> Path | None:
    """Combine vx + vz into one long CSV per (time, node, x, z, vx, vz)."""
    if not vx_path.exists() or not vz_path.exists():
        return None
    out = vx_path.parent / "velocity.csv"

    def _read_field(p: Path) -> dict[tuple[float, int], tuple[float, float, float]]:
        # (t, n) -> (x, z, value)
        result: dict[tuple[float, int], tuple[float, float, float]] = {}
        current_t = None
        num_np: int | None = None
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            m = re.match(r"Time\s*\*\*\*\s*([0-9.eE+\-]+)", s)
            if m:
                current_t = float(m.group(1)); continue
            if current_t is None:
                continue
            toks = s.split()
            if not toks:
                continue
            try:
                n = int(toks[0])
            except ValueError:
                continue
            if len(toks) < 5:
                continue
            try:
                x = float(toks[1]); z = float(toks[2])
                v1 = float(toks[3]); v2 = float(toks[4])
            except ValueError:
                continue
            result[(current_t, n)] = (x, z, v1)
            result[(current_t, n + 1)] = (x, z, v2)
        return result

    a = _read_field(vx_path)
    b = _read_field(vz_path)
    keys = sorted(set(a.keys()) & set(b.keys()))
    rows = [
        [t, n, a[(t, n)][0], a[(t, n)][1], a[(t, n)][2], b[(t, n)][2]]
        for (t, n) in keys
    ]
    _write_csv(out, ["time", "node", "x", "z", "vx", "vz"], rows)
    return out


# ----------------------------------------------------------------------
# Richards3D: VTU series → long CSV
# ----------------------------------------------------------------------

def convert_vtu_series(pvd_path: Path) -> Path | None:
    """Convert a .pvd series + its .vtu frames to mesh_long.csv.

    Requires meshio for fast parsing; without it we skip (don't crash).
    """
    try:
        import meshio
    except ImportError:
        return None
    if not pvd_path.exists():
        return None
    # Parse .pvd to get (t, file) pairs
    import xml.etree.ElementTree as ET
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    pairs: list[tuple[float, Path]] = []
    for ds in root.iter("DataSet"):
        t = float(ds.get("timestep", "0"))
        f = ds.get("file", "")
        pairs.append((t, pvd_path.parent / f))
    if not pairs:
        return None
    out = pvd_path.with_name(pvd_path.stem + "_long.csv")
    rows: list[list] = []
    header_extras: list[str] = []
    for t, fp in pairs:
        m = meshio.read(str(fp))
        if not header_extras:
            header_extras = sorted(m.point_data.keys())
        pts = m.points  # (N, 3)
        for n in range(len(pts)):
            row = [t, n + 1, pts[n][0], pts[n][1], pts[n][2]]
            for k in header_extras:
                row.append(float(m.point_data[k][n]))
            rows.append(row)
    _write_csv(out, ["time", "node", "x", "y", "z", *header_extras], rows)
    return out


# ----------------------------------------------------------------------
# Top-level: walk a directory, convert everything we recognise
# ----------------------------------------------------------------------

def convert_output_dir(out_dir: Path | str,
                        grid_path: Path | None = None) -> list[Path]:
    """Scan `out_dir`, convert every recognised .OUT/.out/.vtu, return
    the list of CSV paths written."""
    d = Path(out_dir)
    if not d.is_dir():
        return []
    written: list[Path] = []
    # HYDRUS-1D outputs
    by_name = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
    if "t_level.out" in by_name:
        written.append(convert_t_level(by_name["t_level.out"]))
    if "balance.out" in by_name:
        # Could be H1D or SWMS — content sniff
        sample = by_name["balance.out"].read_text(errors="ignore")[:200]
        if "WatBalT" in sample or "WatBalR" in sample:
            written.append(convert_balance_swms(by_name["balance.out"]))
        else:
            written.append(convert_balance_h1d(by_name["balance.out"]))
    if "nod_inf.out" in by_name:
        written.append(convert_nod_inf(by_name["nod_inf.out"]))
    # SWMS_2D outputs
    if "run_inf.out" in by_name:
        written.append(convert_run_inf(by_name["run_inf.out"]))
    if "h.out" in by_name:
        written.append(convert_swms_field(by_name["h.out"], "h", grid_path))
    if "th.out" in by_name:
        written.append(convert_swms_field(by_name["th.out"], "theta", grid_path))
    if "q.out" in by_name:
        written.append(convert_swms_field(by_name["q.out"], "q", grid_path))
    vx, vz = by_name.get("vx.out"), by_name.get("vz.out")
    if vx and vz:
        v = convert_swms_velocity(vx, vz, grid_path)
        if v: written.append(v)
    # Richards3D .pvd series
    for pvd in d.glob("*.pvd"):
        v = convert_vtu_series(pvd)
        if v: written.append(v)
    return [p for p in written if p]
