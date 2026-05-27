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


def _detect_conc_col(header_line: str) -> int | None:
    """Return 0-based column index of the first 'Conc' column in a NOD_INF
    header line, or None if not present.  The header usually looks like:
    ``Node   Depth   Head   Moisture   K   C   Flux   Sink   Kappa   v/KsTop   Temp   C1``
    where 'C1' (or 'C', 'Conc') is the solute concentration column.
    The HYDRUS hydraulic-capacity column is also labelled 'C' (position 5
    in the standard layout); we distinguish the *solute* column as the one
    AFTER 'Temp' (the temperature column) in the extended header format.
    """
    tokens = header_line.split()
    # Look for Temp column first; the first Cx token after it is the solute.
    temp_idx = None
    for i, tok in enumerate(tokens):
        if tok.upper() == "TEMP":
            temp_idx = i
            break
    if temp_idx is not None:
        for i in range(temp_idx + 1, len(tokens)):
            tok = tokens[i]
            if tok.upper().startswith("C") and tok.upper() not in ("[L]", "[M/L3]", "[M/L³]", "[C]"):
                return i
    # Fallback: look for explicit Conc/Conc1 labels
    for i, tok in enumerate(tokens):
        if tok.upper() in ("CONC", "CONC1"):
            return i
    return None


def parse_conc_from_nod_inf(path: str | Path) -> np.ndarray:
    """Parse the N-NO₃ concentration field from NOD_INF.OUT.

    Returns
    -------
    np.ndarray, shape (nT, nZ)
        Concentration [mg/L] at each (time, depth) node.  Depth axis is
        **ascending positive** (same convention as :func:`parse_nod_inf`).
        If no 'C' column exists (non-solute run), raises ``KeyError`` which
        the caller should catch and fall back to zeros.
    """
    text = Path(path).read_text()
    blocks = re.split(r"^\s*Time:\s*", text, flags=re.MULTILINE)[1:]
    times: list[float] = []
    conc_rows: list[np.ndarray] = []
    z_descending: np.ndarray | None = None
    conc_col: int | None = None

    for blk in blocks:
        lines = blk.splitlines()
        if not lines:
            continue
        t_val = float(lines[0].split()[0])

        # Locate the header line (contains non-numeric tokens like "Node")
        header_idx = None
        for li, line in enumerate(lines[1:], start=1):
            if line.strip() and not line.strip()[0].lstrip("-").isdigit():
                header_idx = li
                if conc_col is None:
                    conc_col = _detect_conc_col(line)
                break

        rows_z: list[float] = []
        rows_c: list[float] = []
        for line in lines[(header_idx + 1 if header_idx is not None else 1):]:
            parts = line.split()
            if len(parts) < 4 or not parts[0].lstrip("-").isdigit():
                continue
            rows_z.append(float(parts[1]))
            # concentration column index; default to 5 (Node,z,h,theta,K,C)
            col = conc_col if conc_col is not None else 5
            rows_c.append(float(parts[col]) if len(parts) > col else 0.0)

        if not rows_z:
            continue
        times.append(t_val)
        if z_descending is None:
            z_descending = np.array(rows_z)
        conc_rows.append(np.array(rows_c))

    if conc_col is None:
        raise KeyError("No concentration column found in NOD_INF.OUT")

    z_desc = np.asarray(z_descending)
    conc = np.array(conc_rows)
    order = np.argsort(np.abs(z_desc))
    conc = conc[:, order]
    return conc


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
