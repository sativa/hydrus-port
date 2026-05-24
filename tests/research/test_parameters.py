import numpy as np
import pytest
from hydrus_research.parameters import ParameterSpec, ParameterMap


def test_parameter_spec_linear_default():
    s = ParameterSpec(name="alpha", target="materials[0].alpha", bounds=(0.001, 1.0))
    assert s.transform == "linear"
    assert s.to_internal(0.5) == 0.5
    assert s.from_internal(0.5) == 0.5


def test_parameter_spec_log_transform():
    s = ParameterSpec(name="Ks", target="materials[0].Ks", bounds=(0.01, 100.0),
                      transform="log")
    internal = s.to_internal(10.0)
    assert internal == pytest.approx(np.log(10.0))
    assert s.from_internal(internal) == pytest.approx(10.0)


def test_parameter_spec_logit_transform():
    s = ParameterSpec(name="frac", target="x", bounds=(0.0, 1.0), transform="logit")
    internal = s.to_internal(0.5)
    assert internal == pytest.approx(0.0)
    assert s.from_internal(0.0) == pytest.approx(0.5)


def test_parameter_spec_internal_bounds():
    s = ParameterSpec(name="Ks", target="x", bounds=(0.01, 100.0), transform="log")
    lo, hi = s.internal_bounds()
    assert lo == pytest.approx(np.log(0.01))
    assert hi == pytest.approx(np.log(100.0))


def test_parameter_spec_rejects_invalid_transform():
    with pytest.raises(ValueError):
        ParameterSpec(name="x", target="x", bounds=(0, 1), transform="cubic")


def _three_specs():
    return [
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(0.001, 1.0), transform="log"),
        ParameterSpec(name="n",     target="materials[0].n",
                      bounds=(1.05, 5.0),  transform="linear"),
        ParameterSpec(name="Ks",    target="materials[0].Ks",
                      bounds=(0.01, 100.0), transform="log"),
    ]


def test_parameter_map_roundtrip():
    pm = ParameterMap(_three_specs())
    named = {"alpha": 0.05, "n": 1.5, "Ks": 10.0}
    theta = pm.to_vector(named)
    back = pm.from_vector(theta)
    for k, v in named.items():
        assert back[k] == pytest.approx(v)


def test_parameter_map_bounds_array_internal():
    pm = ParameterMap(_three_specs())
    bnds = pm.bounds_array()                 # shape (D, 2), internal coords
    assert bnds.shape == (3, 2)
    assert bnds[0, 0] == pytest.approx(np.log(0.001))
    assert bnds[1, 0] == pytest.approx(1.05)


def test_parameter_map_midpoints():
    pm = ParameterMap(_three_specs())
    mids = pm.midpoints()
    assert mids.shape == (3,)
    # n is linear: midpoint of (1.05, 5.0) is 3.025
    assert mids[1] == pytest.approx(3.025)


def test_parameter_map_requires_unique_names():
    dup = [
        ParameterSpec(name="x", target="a", bounds=(0, 1)),
        ParameterSpec(name="x", target="b", bounds=(0, 1)),
    ]
    with pytest.raises(ValueError):
        ParameterMap(dup)


def test_apply_to_scenario_patches_dict_path():
    pm = ParameterMap(_three_specs())
    scenario = {
        "materials": [
            {"alpha": 0.01, "n": 1.4, "Ks": 1.0},
            {"alpha": 0.05, "n": 1.5, "Ks": 5.0},
        ],
    }
    patched = pm.apply_to_scenario(scenario, {"alpha": 0.123, "n": 2.0, "Ks": 50.0})
    assert patched["materials"][0]["alpha"] == 0.123
    assert patched["materials"][0]["n"] == 2.0
    assert patched["materials"][0]["Ks"] == 50.0
    # original untouched (we return a deep copy)
    assert scenario["materials"][0]["alpha"] == 0.01
    # other-index material untouched
    assert patched["materials"][1]["alpha"] == 0.05


def test_apply_to_scenario_supports_nested_path():
    pm = ParameterMap([
        ParameterSpec(name="tol", target="solver.tol_theta", bounds=(1e-6, 1e-2)),
    ])
    scenario = {"solver": {"tol_theta": 0.001, "max_picard": 20}}
    patched = pm.apply_to_scenario(scenario, {"tol": 0.005})
    assert patched["solver"]["tol_theta"] == 0.005
    assert patched["solver"]["max_picard"] == 20


def test_apply_to_scenario_rejects_unknown_path():
    pm = ParameterMap([
        ParameterSpec(name="x", target="does.not.exist", bounds=(0, 1)),
    ])
    with pytest.raises(KeyError):
        pm.apply_to_scenario({"materials": []}, {"x": 0.5})


def test_apply_to_scenario_roundtrip_with_real_scenario():
    """The patched dict must round-trip through hydrus_port.schema, proving
    the path format is compatible with the canonical schema."""
    from hydrus_port.schema import Scenario, ScenarioMeta, Units, Solver, \
        HydraulicMaterial, TimeControl, Geometry1D, _scenario_from_dict
    s = Scenario(
        meta=ScenarioMeta(name="t"),
        units=Units(),
        solver=Solver(),
        materials=[HydraulicMaterial(theta_r=0.05, theta_s=0.4,
                                     alpha=0.02, n=1.5, Ks=10.0)],
        time=TimeControl(t_init=0.0, t_max=1.0),
        geometry=Geometry1D(z=[0.0, 50.0, 100.0],
                            initial_h=[-100.0, -100.0, -100.0],
                            mat_num=[1, 1, 1]),
    )
    d = s.to_dict()
    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha", bounds=(0.001, 1.0)),
    ])
    patched = pm.apply_to_scenario(d, {"alpha": 0.099})
    s2 = _scenario_from_dict(patched)
    assert s2.materials[0].alpha == 0.099
    # original Scenario object untouched
    assert s.materials[0].alpha == 0.02
