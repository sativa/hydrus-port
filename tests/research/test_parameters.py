import numpy as np
import pytest
from hydrus_research.parameters import ParameterSpec


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
