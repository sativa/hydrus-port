"""PEST++ per-realization worker entry point.

Invoked by PESTPP-IES as: `python -m hydrus_research.inversion.worker_script
<theta.dat> <y.dat> <config.json>`.

The script reads:
  - theta.dat   — one float per line, internal coords, in param-spec order
  - config.json — {scenario_dir, param_specs[], obs_specs[]}

Rebuilds `make_forward(...)` locally, runs it, writes y.dat (one float per
line, in obs-spec order).

No Python-object serialization crosses the process boundary — only JSON
config and plain-text numerical I/O. This is the contract PEST++ workers
operate under."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def _run(theta_path: Path, y_path: Path, cfg_path: Path) -> int:
    cfg = json.loads(cfg_path.read_text())
    scenario_dir = Path(cfg["scenario_dir"])
    param_specs_json = cfg["param_specs"]
    obs_specs_json = cfg["obs_specs"]

    # Local imports — the worker pulls in the full engine when invoked
    from hydrus_research.parameters import ParameterSpec, ParameterMap
    from hydrus_research.observations import ObservationSpec
    from hydrus_research.simulator import make_forward
    from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
    from hydrus_port.adapters.hydrus1d import load as _load_h1d

    specs = [ParameterSpec(**s) for s in param_specs_json]
    pm = ParameterMap(specs)
    obs_specs = [ObservationSpec(**o) for o in obs_specs_json]
    template = _load_h1d(scenario_dir).to_dict()
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)

    theta = np.array([float(x) for x in theta_path.read_text().split()])
    if theta.shape != (len(specs),):
        print(f"theta.dat has {theta.shape[0]} values, expected {len(specs)}",
              file=sys.stderr)
        return 2

    y = np.asarray(forward(theta), dtype=float)
    y_path.write_text("\n".join(f"{v:.10e}" for v in y))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3:
        print("usage: worker_script <theta.dat> <y.dat> <config.json>",
              file=sys.stderr)
        return 64
    return _run(Path(argv[0]), Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main())
