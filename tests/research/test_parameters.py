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
