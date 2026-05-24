import pytest
from hydrus_research.ptf.carsel_parrish import (
    USDA_CLASSES, carsel_parrish_lookup,
)


def test_loam_matches_1988_table():
    r = carsel_parrish_lookup("loam")
    # Carsel & Parrish 1988 Table 2: loam mean values
    assert r.theta_r == pytest.approx(0.078, abs=0.001)
    assert r.theta_s == pytest.approx(0.43,  abs=0.005)
    assert r.alpha   == pytest.approx(0.036, abs=0.002)
    assert r.n       == pytest.approx(1.56,  abs=0.02)
    assert r.Ks      == pytest.approx(24.96, abs=0.1)   # cm/day
    assert r.method  == "carsel_parrish"


def test_sand_matches_1988_table():
    r = carsel_parrish_lookup("sand")
    assert r.theta_r == pytest.approx(0.045, abs=0.001)
    assert r.theta_s == pytest.approx(0.43,  abs=0.005)
    assert r.alpha   == pytest.approx(0.145, abs=0.005)
    assert r.n       == pytest.approx(2.68,  abs=0.05)
    assert r.Ks      == pytest.approx(712.8, abs=1.0)


def test_clay_matches_1988_table():
    r = carsel_parrish_lookup("clay")
    assert r.theta_r == pytest.approx(0.068, abs=0.002)
    assert r.theta_s == pytest.approx(0.38,  abs=0.005)
    assert r.alpha   == pytest.approx(0.008, abs=0.001)
    assert r.n       == pytest.approx(1.09,  abs=0.02)
    assert r.Ks      == pytest.approx(4.8,   abs=0.1)


def test_all_12_classes_present():
    assert len(USDA_CLASSES) == 12
    for c in ("sand", "loamy_sand", "sandy_loam", "loam", "silt", "silt_loam",
              "sandy_clay_loam", "clay_loam", "silty_clay_loam",
              "sandy_clay", "silty_clay", "clay"):
        assert c in USDA_CLASSES


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        carsel_parrish_lookup("not_a_texture")
