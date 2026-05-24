import json
import pytest
from datetime import date
from pathlib import Path

from hydrus_research.dndc_seam.adapters import (
    DndcSeamAdapter, ManualFileAdapter, DndcLiveAdapter,
)
from hydrus_research.dndc_seam.schema import DndcSeamInputs


def _write_minimal_json(path: Path):
    di = {
        "atm": {"dates": ["2026-05-01"], "precip_cm": [0.0], "pet_cm": [0.4]},
        "et":  {"mode": "lai_beer", "lai": [2.0], "extinction_k": 0.6},
        "root": {"z_max_cm": 50, "growth_curve": "logistic", "days_to_zmax": 30,
                 "density_profile": "linear_decline"},
        "feddes": {"h1": -15, "h2": -30, "h3_high": -325, "h3_low": -600, "h4": -8000,
                   "pet_high_cm_d": 0.5, "pet_low_cm_d": 0.1},
        "n_transform": {"mode": "constant_rates", "k_nitrification_d": 0.1},
        "plant_n_uptake": {"mode": "passive_with_water"},
        "state": {"z_grid_cm": [0.0]},
    }
    path.write_text(json.dumps(di))


def test_manual_file_adapter_loads_json(tmp_path):
    p = tmp_path / "preset.json"
    _write_minimal_json(p)
    ad = ManualFileAdapter(p)
    out = ad.produce("any_scenario", (date(2026, 5, 1), date(2026, 5, 2)))
    assert isinstance(out, DndcSeamInputs)
    assert out.atm.precip_cm == [0.0]


def test_dndc_live_adapter_is_subclass_of_seam_adapter():
    assert issubclass(DndcLiveAdapter, DndcSeamAdapter)


def test_dndc_live_adapter_raises_until_b2():
    ad = DndcLiveAdapter(dndc_session=None)
    with pytest.raises(NotImplementedError):
        ad.produce("s", (date(2026, 5, 1), date(2026, 5, 2)))


def test_seam_adapter_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DndcSeamAdapter()
