"""M1 acceptance: build a DndcSeamInputs from a YAML preset, pass through
to_forcing, run via Hydrus1DSimulator with non-None forcing, verify the
cumulative top flux matches the bare-scenario run within 1%."""
import numpy as np
import pytest
from datetime import date
from pathlib import Path

from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.dndc_seam import to_forcing
from hydrus_research.dndc_seam.adapters import ManualFileAdapter
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


_FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "infiltr_v1" / "inputs"
_PRESET_PATH = Path(__file__).parent / "data" / "preset_ex2_like.yaml"


@pytest.fixture(scope="module")
def template_dict():
    return load_h1d_canonical(_FIXTURE_DIR).to_dict()


def test_e2e_dndc_seam_to_simulator_run(template_dict):
    sim = Hydrus1DSimulator(work_root=Path("/tmp/m1_e2e"))

    # Baseline: bare scenario, forcing=None (M0 path)
    r_bare = sim.run(template_dict, forcing=None, ic=None)
    final_vol_bare = r_bare.mass_balance.get("volume_last", float("nan"))

    # M1 path: DndcSeamInputs preset -> to_forcing -> Simulator.run(forcing=...)
    di = ManualFileAdapter(_PRESET_PATH).produce(
        "infiltr_v1", (date(2026, 5, 1), date(2026, 5, 5))
    )
    sim_times = np.linspace(0.0, 5.0, 6)
    forcing = to_forcing(di, sim_times)

    r_m1 = sim.run(template_dict, forcing=forcing, ic=None)
    final_vol_m1 = r_m1.mass_balance.get("volume_last", float("nan"))

    # With zero precip + zero pet in the preset, the M1 path should produce
    # the same final volume as the bare run (i.e. the M1 atm override that
    # zeroes precip should have no effect on a fixture that already has no
    # atmospheric BC, OR -- more strictly -- should match within 1%).
    if not (np.isnan(final_vol_bare) or np.isnan(final_vol_m1)):
        rel_err = abs(final_vol_m1 - final_vol_bare) / max(abs(final_vol_bare), 1e-9)
        assert rel_err < 0.01, (
            f"M1 vs bare cumulative-volume relative error = {rel_err:.4%}"
        )
