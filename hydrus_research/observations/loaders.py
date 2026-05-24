"""Loaders that build an ObservationSet from existing HYDRUS / SWMS output."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np

from .spec import ObservationSpec
from .set import ObservationSet


# OBS_NODE.OUT layout (HYDRUS-1D 4.08):
#   banner lines (start with ' *') then blank lines
#   one header line listing per-node block names: "Node( 1)  Node( 21) ..."
#   one sub-header listing per-node column names: "time  h  theta  Temp  h  theta  Temp ..."
#   data rows: "time   h1 theta1 [T1]   h2 theta2 [T2] ..."
# The exact spacing varies between Fortran builds, so we tokenize on whitespace.

_NODE_RE = re.compile(r"Node\s*\(\s*(\d+)\s*\)")


def _parse_obsnod(path: Path) -> tuple[list[int], list[str], np.ndarray]:
    """Returns (node_ids, per_node_columns, data_array).

    data_array has shape (NT, 1 + n_nodes * n_cols) where col 0 is time."""
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]
    # find the line listing Node(N) tokens
    node_line_idx = None
    for i, ln in enumerate(lines):
        if _NODE_RE.search(ln):
            node_line_idx = i
            break
    if node_line_idx is None:
        raise ValueError(f"no 'Node(N)' header found in {path}")
    node_ids = [int(m.group(1)) for m in _NODE_RE.finditer(lines[node_line_idx])]

    # the next non-comment, non-empty line is the per-node column header
    col_idx = node_line_idx + 1
    while col_idx < len(lines) and (lines[col_idx].startswith("#") or not lines[col_idx].strip()):
        col_idx += 1
    header_tokens = lines[col_idx].split()
    # tokens look like: time h theta Temp h theta Temp ...
    # n_cols_per_node = (len(header_tokens) - 1) // len(node_ids)
    n_nodes = len(node_ids)
    n_cols = (len(header_tokens) - 1) // n_nodes
    per_node_cols = header_tokens[1 : 1 + n_cols]   # ["h", "theta", "Temp"?]

    # data rows
    data_rows: list[list[float]] = []
    for ln in lines[col_idx + 1 :]:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("end"):
            continue
        try:
            data_rows.append([float(x) for x in s.split()])
        except ValueError:
            break    # ran past the numeric section
    return node_ids, per_node_cols, np.array(data_rows, dtype=float)


def from_hydrus_obsnod(path: Path | str,
                       kinds: tuple[str, ...] = ("theta",),
                       times_day: list[float] | None = None,
                       default_sigma: dict[str, float] | None = None
                       ) -> ObservationSet:
    """Build an ObservationSet from a HYDRUS-1D OBS_NODE.OUT file.

    Parameters
    ----------
    path : OBS_NODE.OUT location.
    kinds : which observable columns to harvest. Choose any of
        {"theta", "h", "c", "T"} that are present in the file.
    times_day : list of times to sample (linear interp on the file's time axis).
        If None, every printed time is used.
    default_sigma : per-kind measurement-error stddev; default 0.01 for theta,
        1.0 for h, 0.5 for c, 0.5 for T.
    """
    path = Path(path)
    node_ids, cols, data = _parse_obsnod(path)
    times = data[:, 0]
    n_nodes = len(node_ids)
    n_cols_per_node = len(cols)
    # column index helper inside one node block (lowercase for robust matching)
    col_pos = {name.lower(): i for i, name in enumerate(cols)}

    sigma_defaults = {"theta": 0.01, "h": 1.0, "c": 0.5, "T": 0.5}
    if default_sigma:
        sigma_defaults.update(default_sigma)

    if times_day is None:
        times_day = list(times)

    specs: list[ObservationSpec] = []
    vals: list[float] = []
    sigs: list[float] = []
    for kind in kinds:
        key = "conc" if kind == "c" else ("temp" if kind == "T" else kind)
        if key not in col_pos:
            raise KeyError(f"requested kind {kind!r} (column {key!r}) not in OBS_NODE.OUT")
        col_in_node = col_pos[key]
        for node_id, node_block in zip(node_ids, range(n_nodes)):
            col_in_data = 1 + node_block * n_cols_per_node + col_in_node
            series = data[:, col_in_data]
            for t in times_day:
                v = float(np.interp(t, times, series))
                specs.append(ObservationSpec(
                    name=f"{kind}_node{node_id}_d{t:g}",
                    kind=kind if kind in ("theta", "h", "c") else "h",  # T not in M0
                    location={"node": node_id},
                    time_day=float(t),
                ))
                vals.append(v)
                sigs.append(sigma_defaults.get(kind, 1.0))
    return ObservationSet(specs=specs,
                          values=np.array(vals),
                          sigmas=np.array(sigs))
