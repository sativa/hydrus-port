"""M0 acceptance: forward(theta) -> y_sim end-to-end on infiltr_v1."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


@pytest.fixture(scope="module")
def template_dict():
    return load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()


def test_forward_theta_to_y_end_to_end(template_dict):
    sim = Hydrus1DSimulator(work_root=Path("/tmp/hres_m0_e2e"))

    alpha0 = template_dict["materials"][0]["alpha"]
    n0     = template_dict["materials"][0]["n"]
    Ks0    = template_dict["materials"][0]["Ks"]

    pm = ParameterMap([
        ParameterSpec(name="alpha",
                      target="materials[0].alpha",
                      bounds=(alpha0 * 0.5, alpha0 * 2.0),
                      transform="log"),
        ParameterSpec(name="n",
                      target="materials[0].n",
                      bounds=(max(1.05, n0 * 0.8), n0 * 1.5),
                      transform="linear"),
        ParameterSpec(name="Ks",
                      target="materials[0].Ks",
                      bounds=(Ks0 * 0.1, Ks0 * 10.0),
                      transform="log"),
    ])

    # Get the available z extent + simulation duration from the template, so
    # we observe at points the run will actually visit.
    z_nodes = template_dict["geometry"]["z"]
    z_min, z_max = min(z_nodes), max(z_nodes)
    t_max = template_dict["time"]["t_max"]

    obs_specs = [
        ObservationSpec(name=f"theta_mid_t_{t:g}",
                        kind="theta",
                        location={"z_cm": 0.5 * (z_min + z_max)},
                        time_day=t)
        for t in np.linspace(0.0, t_max, 5)[1:]    # drop t=0
    ]

    forward = make_forward(sim, pm,
                           template_scenario=template_dict,
                           forcing=None, ic=None,
                           obs_specs=obs_specs)

    # Reference run at nominal parameters
    theta_ref = pm.to_vector({"alpha": alpha0, "n": n0, "Ks": Ks0})
    y_ref = forward(theta_ref)

    # Perturbed run: increase alpha by 20%
    theta_pert = pm.to_vector({"alpha": alpha0 * 1.2, "n": n0, "Ks": Ks0})
    y_pert = forward(theta_pert)

    # ---- acceptance checks ----
    assert y_ref.shape == (len(obs_specs),)
    assert np.all(np.isfinite(y_ref))
    assert np.all((y_ref >= 0) & (y_ref < 1.0)), "theta out of physical range"
    # Parameter perturbation must change at least one observable
    assert not np.allclose(y_ref, y_pert), \
        "alpha perturbation did not change any observable; abstraction is broken"
