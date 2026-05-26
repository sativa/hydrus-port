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
        t_val = float(head.split()[0])
        rows = []
        for line in rest:
            parts = line.split()
            if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
                continue
            rows.append([float(x) for x in parts[:4]])
        if not rows:
            continue
        times.append(t_val)
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
    """Parse HYDRUS-1D BALANCE.OUT into mm-units agronomy water-balance dict."""
    p = Path(path)
    totals = {"rain_mm": 0.0, "et_mm": 0.0,
              "percolation_mm": 0.0, "storage_change_mm": 0.0}
    if not p.exists():
        return totals
    text = p.read_text()

    block_re = re.compile(r"Time\s+\[T\]\s+([\-0-9.E+]+)")
    field_re = {
        "W":    re.compile(r"W-volume\s+\[L\]\s+([\-0-9.E+]+)"),
        "Ftop": re.compile(r"Top Flux\s+\[L/T\]\s+([\-0-9.E+]+)"),
        "Fbot": re.compile(r"Bot Flux\s+\[L/T\]\s+([\-0-9.E+]+)"),
    }

    # Split on Time [T] markers; each chunk starting at index 1 is a block.
    # Use re.split with a capturing group so we get the time value too.
    raw_blocks = re.split(r"(?m)^.*Time\s+\[T\]\s+([\-0-9.E+]+).*$", text)
    # raw_blocks alternates: [pre-text, t0, chunk0, t1, chunk1, ...]
    blocks = []
    for i in range(1, len(raw_blocks) - 1, 2):
        t_str = raw_blocks[i]
        chunk = raw_blocks[i + 1]
        rec: dict[str, float] = {"t": float(t_str)}
        for k, rx in field_re.items():
            mm = rx.search(chunk)
            if mm:
                rec[k] = float(mm.group(1))
        blocks.append(rec)

    if len(blocks) < 2:
        return totals

    # Storage change (cm -> mm)
    w0 = blocks[0].get("W")
    w1 = blocks[-1].get("W")
    if w0 is not None and w1 is not None:
        totals["storage_change_mm"] = (w1 - w0) * 10.0

    # Trapezoidal integration of fluxes (cm/day * day = cm -> mm)
    rain_cm = et_cm = perc_cm = 0.0
    for a, b in zip(blocks[:-1], blocks[1:]):
        if "t" not in a or "t" not in b:
            continue
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        for f_name, sign_target, accumulator in (
            ("Ftop", +1, "rain"),
            ("Ftop", -1, "et"),
            ("Fbot", -1, "perc"),
        ):
            fa = a.get(f_name)
            fb = b.get(f_name)
            if fa is None or fb is None:
                continue
            avg = 0.5 * (fa + fb)
            if sign_target > 0 and avg > 0:
                rain_cm += avg * dt
            elif sign_target < 0 and avg < 0:
                if accumulator == "et":
                    et_cm += abs(avg) * dt
                else:
                    perc_cm += abs(avg) * dt

    totals["rain_mm"] += rain_cm * 10.0
    totals["et_mm"]    += et_cm   * 10.0
    totals["percolation_mm"] += perc_cm * 10.0

    return totals
