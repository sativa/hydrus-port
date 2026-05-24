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


from hydrus_research.dndc_seam.schema import FertEvent, IrrigEvent


def test_fert_event_surface_urea():
    f = FertEvent(date=date(2026, 5, 15), depth_cm=0.0,
                  mass_kg_n_ha=120.0, form="urea")
    assert f.composition is None


def test_fert_event_compound_requires_composition():
    with pytest.raises(ValidationError):
        FertEvent(date=date(2026, 5, 15), depth_cm=0.0,
                  mass_kg_n_ha=120.0, form="compound")    # missing composition


def test_fert_event_compound_with_composition():
    f = FertEvent(date=date(2026, 5, 15), depth_cm=5.0,
                  mass_kg_n_ha=120.0, form="compound",
                  composition={"NH4": 0.5, "NO3": 0.5})
    assert sum(f.composition.values()) == pytest.approx(1.0)


def test_irrig_event_drip_with_fertigation():
    i = IrrigEvent(date=date(2026, 6, 1), method="drip",
                   amount_cm=0.5, duration_h=2.0,
                   solute_concs_mg_l={"NO3": 80.0},
                   drip_emitter_xyz=(10.0, 10.0, -5.0))
    assert i.method == "drip"
    assert i.solute_concs_mg_l == {"NO3": 80.0}


def test_irrig_event_flood_no_xyz():
    i = IrrigEvent(date=date(2026, 6, 1), method="flood",
                   amount_cm=5.0, duration_h=12.0)
    assert i.drip_emitter_xyz is None
    assert i.solute_concs_mg_l == {}


from hydrus_research.dndc_seam.schema import NTransformation, PlantNUptake


def test_n_transform_constant_rates():
    n = NTransformation(mode="constant_rates",
                        k_mineralization_d=0.005,
                        k_nitrification_d=0.1,
                        k_denitrification_d=0.02,
                        k_volatilization_d=0.01)
    assert n.mode == "constant_rates"


def test_n_transform_constant_requires_rates():
    with pytest.raises(ValidationError):
        NTransformation(mode="constant_rates")      # no rates given


def test_n_transform_external_callable_b2_hook():
    n = NTransformation(mode="external_callable",
                        callable_ref="dndc.n_module:compute_rates")
    assert n.callable_ref == "dndc.n_module:compute_rates"


def test_n_transform_external_requires_ref():
    with pytest.raises(ValidationError):
        NTransformation(mode="external_callable")


def test_plant_n_uptake_michaelis_menten():
    p = PlantNUptake(mode="michaelis_menten",
                     km_mg_l=10.0,
                     vmax_mg_per_day_per_root_cm=0.05)
    assert p.km_mg_l == 10.0


def test_plant_n_uptake_demand_requires_daily_demand():
    with pytest.raises(ValidationError):
        PlantNUptake(mode="demand_driven")


from hydrus_research.dndc_seam.schema import (
    StateExchange, SoilTemp, Residue, DndcSeamInputs,
)


def test_state_exchange_minimal():
    s = StateExchange(z_grid_cm=[0.0, -10.0, -50.0, -100.0])
    assert s.writeback_daily is False
    assert s.initial_theta is None


def test_state_exchange_initial_profile_length_must_match():
    with pytest.raises(ValidationError):
        StateExchange(z_grid_cm=[0.0, -10.0],
                      initial_theta=[0.3, 0.32, 0.34])      # 3 vs 2 nodes


def test_soil_temp_disabled_default():
    s = SoilTemp()
    assert s.enabled is False


def test_residue_defaults_zero():
    r = Residue()
    assert r.mulch_fraction == 0.0
    assert r.e_reduction_factor == 1.0


def test_dndc_seam_inputs_parent_assembly():
    """Build a minimal-valid DndcSeamInputs from the sub-models."""
    di = DndcSeamInputs(
        atm=AtmDaily(dates=[date(2026, 5, 1), date(2026, 5, 2)],
                     precip_cm=[0.0, 0.3], pet_cm=[0.4, 0.5]),
        et=EtPartition(mode="lai_beer", lai=[2.0, 2.2]),
        root=RootGrowth(z_max_cm=50, growth_curve="logistic", days_to_zmax=30),
        feddes=FeddesParams(h1=-15, h2=-30, h3_high=-325, h3_low=-600, h4=-8000),
        n_transform=NTransformation(mode="constant_rates", k_nitrification_d=0.1),
        plant_n_uptake=PlantNUptake(mode="passive_with_water"),
        state=StateExchange(z_grid_cm=[0.0, -10.0, -50.0]),
    )
    assert di.fert_events == []
    assert di.irrig_events == []
    assert di.extras == {}


def test_dndc_seam_inputs_round_trips_through_json():
    di = DndcSeamInputs(
        atm=AtmDaily(dates=[date(2026, 5, 1)], precip_cm=[0.0]),
        et=EtPartition(mode="lai_beer", lai=[2.0]),
        root=RootGrowth(z_max_cm=50, growth_curve="logistic", days_to_zmax=30),
        feddes=FeddesParams(h1=-15, h2=-30, h3_high=-325, h3_low=-600, h4=-8000),
        n_transform=NTransformation(mode="constant_rates", k_nitrification_d=0.1),
        plant_n_uptake=PlantNUptake(mode="passive_with_water"),
        state=StateExchange(z_grid_cm=[0.0]),
    )
    js = di.model_dump_json()
    di2 = DndcSeamInputs.model_validate_json(js)
    assert di2.atm.precip_cm == [0.0]
    assert di2.root.z_max_cm == 50
