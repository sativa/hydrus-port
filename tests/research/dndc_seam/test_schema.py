import pytest
from datetime import date
from pydantic import ValidationError

from hydrus_research.dndc_seam.schema import (
    AtmDaily, EtPartition, RootGrowth, FeddesParams,
)


def test_atm_daily_minimum():
    a = AtmDaily(dates=[date(2026, 5, 1), date(2026, 5, 2)],
                 precip_cm=[0.0, 0.3])
    assert len(a.dates) == 2
    assert a.pet_cm is None         # optional


def test_atm_daily_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        AtmDaily(dates=[date(2026, 5, 1)],
                 precip_cm=[0.0, 0.3])    # 1 date, 2 precips


def test_et_partition_lai_beer():
    e = EtPartition(mode="lai_beer", lai=[2.0, 2.5, 2.6], extinction_k=0.55)
    assert e.mode == "lai_beer"
    assert e.extinction_k == 0.55
    assert e.kcb is None


def test_et_partition_rejects_kc_dual_without_kcb():
    with pytest.raises(ValidationError):
        EtPartition(mode="kc_dual")        # kcb required for kc_dual mode


def test_root_growth_logistic():
    r = RootGrowth(z_max_cm=80.0, growth_curve="logistic", days_to_zmax=45,
                   density_profile="linear_decline")
    assert r.z_max_cm == 80.0
    assert r.density_param is None


def test_root_growth_rejects_table_without_table():
    with pytest.raises(ValidationError):
        RootGrowth(z_max_cm=80, growth_curve="table",
                   density_profile="linear_decline")


def test_feddes_params_defaults_for_maize():
    f = FeddesParams(h1=-15.0, h2=-30.0, h3_high=-325.0, h3_low=-600.0, h4=-8000.0,
                     pet_high_cm_d=0.5, pet_low_cm_d=0.1)
    assert f.h4 == -8000.0
