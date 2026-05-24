import pytest
from hydrus_research.observations import ObservationSpec


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
