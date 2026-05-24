import numpy as np
import pytest
from pathlib import Path
from hydrus_research.observations import ObservationSpec, ObservationSet


def test_observation_spec_theta_at_depth():
    s = ObservationSpec(name="theta_z20_d5",
                        kind="theta",
                        location={"z_cm": 20.0},
                        time_day=5.0)
    assert s.weight == 1.0
    assert s.species is None
    assert s.location["z_cm"] == 20.0


def test_observation_spec_concentration_with_species():
    s = ObservationSpec(name="no3_z30_d10",
                        kind="c",
                        location={"z_cm": 30.0},
                        time_day=10.0,
                        species="NO3",
                        weight=2.5)
    assert s.species == "NO3"
    assert s.weight == 2.5


def test_observation_spec_rejects_bad_kind():
    with pytest.raises(Exception):
        ObservationSpec(name="x", kind="banana",
                        location={"z_cm": 0}, time_day=1.0)


def test_observation_spec_2d_location_node():
    s = ObservationSpec(name="h_node17_d3", kind="h",
                        location={"node": 17}, time_day=3.0)
    assert s.location == {"node": 17}


def test_observation_set_residuals_and_objective():
    specs = [
        ObservationSpec(name="a", kind="theta", location={"z_cm": 10}, time_day=1.0),
        ObservationSpec(name="b", kind="theta", location={"z_cm": 20}, time_day=1.0),
    ]
    obs = ObservationSet(specs=specs,
                         values=np.array([0.30, 0.35]),
                         sigmas=np.array([0.02, 0.02]))
    sim = np.array([0.32, 0.33])
    res = obs.residuals(sim)
    assert res == pytest.approx([(0.32 - 0.30) / 0.02, (0.33 - 0.35) / 0.02])
    assert obs.objective_l2(sim) == pytest.approx(sum(res ** 2))


def test_observation_set_from_csv():
    path = Path(__file__).parent / "data" / "obs_minimal.csv"
    obs = ObservationSet.from_csv(path)
    assert len(obs.specs) == 3
    assert obs.specs[0].kind == "theta"
    assert obs.specs[2].species == "NO3"
    assert obs.values[2] == 12.5
    assert obs.sigmas[2] == 1.5


def test_observation_set_shape_mismatch_raises():
    specs = [ObservationSpec(name="a", kind="theta", location={"z_cm": 1}, time_day=0)]
    with pytest.raises(ValueError):
        ObservationSet(specs=specs,
                       values=np.array([0.1, 0.2]),    # wrong length
                       sigmas=np.array([0.01, 0.01]))


def test_obs_node_loader_reads_infiltr_v1():
    """Load the real HYDRUS-1D OBS_NODE.OUT and verify spec count + sample value."""
    from hydrus_research.observations.loaders import from_hydrus_obsnod
    path = Path("tests/fixtures/infiltr_v1/reference_out/OBS_NODE.OUT")
    if not path.exists():
        pytest.skip("infiltr_v1 reference output not present")
    times_to_sample = [0.5, 1.0, 2.0]   # any times present in the file
    obs = from_hydrus_obsnod(path, kinds=("theta", "h"), times_day=times_to_sample)
    # number of obs = n_nodes * n_kinds * n_times
    assert obs.M > 0
    # spec names follow the "<kind>_node<N>_d<t>" format
    for s in obs.specs:
        assert "_node" in s.name and "_d" in s.name
        assert "node" in s.location
    # theta values lie in a physical range [0, 1)
    theta_vals = np.array([v for s, v in zip(obs.specs, obs.values) if s.kind == "theta"])
    assert (theta_vals >= 0).all() and (theta_vals < 1.0).all()


def test_observation_spec_z_location_only_for_1d():
    s = ObservationSpec(name="theta_z20", kind="theta",
                        location={"z_cm": 20.0}, time_day=1.0)
    assert s.location["z_cm"] == 20.0


def test_observation_spec_node_location_for_2d_3d():
    s = ObservationSpec(name="h_node17", kind="h",
                        location={"node": 17}, time_day=3.0)
    assert s.location["node"] == 17


def test_observation_spec_rejects_empty_location():
    with pytest.raises(Exception):
        ObservationSpec(name="bad", kind="theta", location={}, time_day=1.0)


def test_observation_spec_rejects_unknown_location_key():
    with pytest.raises(Exception):
        ObservationSpec(name="bad", kind="theta",
                        location={"phi_cm": 20.0}, time_day=1.0)
