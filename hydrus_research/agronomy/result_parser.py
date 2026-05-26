"""Parse HYDRUS-1D output files into agronomy result arrays.

NOD_INF.OUT z is descending (surface 0 -> negative downward). We reverse
z AND every field together so downstream consumers see ascending positive
depths. See memory feedback_hydrus1d_nod_inf_z_descending.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np


def parse_nod_inf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_cm_ascending_positive, t_days, theta_zt[nT, nZ])."""
    text = Path(path).read_text()
    blocks = re.split(r"^\s*Time:\s*", text, flags=re.MULTILINE)[1:]
    times, theta_rows, z_descending = [], [], None
    for blk in blocks:
        head, *rest = blk.splitlines()
        times.append(float(head.split()[0]))
        rows = []
        for line in rest:
            parts = line.split()
            if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
                continue
            rows.append([float(x) for x in parts[:4]])
        arr = np.array(rows)
        if z_descending is None:
            z_descending = arr[:, 1]
        theta_rows.append(arr[:, 3])

    z_desc = np.asarray(z_descending)
    theta = np.array(theta_rows)
    # Reverse so z is ascending; positive depths.
    order = np.argsort(np.abs(z_desc))
    z_pos = np.abs(z_desc[order])
    theta = theta[:, order]
    return z_pos, np.array(times), theta


def parse_balance(path: str | Path) -> dict[str, float]:
    """Read BALANCE.OUT totals -- robust against missing fields."""
    text = Path(path).read_text() if Path(path).exists() else ""
    totals = {"rain_mm": 0.0, "et_mm": 0.0, "percolation_mm": 0.0,
              "storage_change_mm": 0.0}
    # HYDRUS BALANCE.OUT uses lines like "CumFlx(T)= -3.21E+01"; we extract
    # any number we recognize. Missing values stay 0.
    for line in text.splitlines():
        if "Atm" in line and "=" in line:
            try:
                totals["rain_mm"] += abs(float(line.split("=")[-1])) * 10  # cm->mm
            except ValueError:
                pass
        elif "Root" in line and "=" in line:
            try:
                totals["et_mm"] += abs(float(line.split("=")[-1])) * 10
            except ValueError:
                pass
        elif "Bot" in line and "=" in line:
            try:
                totals["percolation_mm"] += abs(float(line.split("=")[-1])) * 10
            except ValueError:
                pass
    return totals
