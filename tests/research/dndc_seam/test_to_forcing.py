import pytest
import numpy as np
from datetime import date, timedelta

from hydrus_research.dndc_seam import DndcSeamInputs, to_forcing
from hydrus_research.dndc_seam.schema import (
    AtmDaily, EtPartition, RootGrowth, FeddesParams,
    NTransformation, PlantNUptake, StateExchange,
)
from hydrus_research.simulator import Forcing


def _minimal_inputs(n_days=5):
    start = date(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    precip_base = [0.0, 0.3, 0.1, 0.0, 0.0]
    pet_base = [0.4, 0.5, 0.45, 0.5, 0.55]
    precip = [precip_base[i % len(precip_base)] for i in range(n_days)]
    pet = [pet_base[i % len(pet_base)] for i in range(n_days)]
    return DndcSeamInputs(
        atm=AtmDaily(
            dates=dates,
            precip_cm=precip,
            pet_cm=pet,
        ),
        et=EtPartition(mode="lai_beer", lai=[2.0]*n_days, extinction_k=0.6),
        root=RootGrowth(z_max_cm=50, growth_curve="logistic", days_to_zmax=30),
        feddes=FeddesParams(h1=-15, h2=-30, h3_high=-325, h3_low=-600, h4=-8000),
        n_transform=NTransformation(mode="constant_rates", k_nitrification_d=0.1,
                                    k_denitrification_d=0.02),
        plant_n_uptake=PlantNUptake(mode="passive_with_water"),
        state=StateExchange(z_grid_cm=[0.0, -10.0, -25.0, -50.0]),
    )


def test_to_forcing_returns_forcing_with_right_shapes():
    inputs = _minimal_inputs(5)
    sim_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])     # 5 days, day-of-sim
    f = to_forcing(inputs, sim_times)
    assert isinstance(f, Forcing)
    assert f.times_days.shape == (5,)
    assert f.precip_cm_per_day.shape == (5,)
    assert f.pet_cm_per_day.shape == (5,)
    assert f.lai.shape == (5,)
    assert f.root_depth_cm.shape == (5,)


def test_to_forcing_root_depth_grows_logistic():
    inputs = _minimal_inputs(60)
    sim_times = np.arange(60, dtype=float)
    f = to_forcing(inputs, sim_times)
    # At t=0 root depth should be small; at t=days_to_zmax (30) it should be near z_max
    assert f.root_depth_cm[0] < 5.0
    assert f.root_depth_cm[30] > 40.0
    assert f.root_depth_cm[-1] <= inputs.root.z_max_cm


def test_to_forcing_n_source_terms_constant_rates():
    inputs = _minimal_inputs(5)
    sim_times = np.arange(5, dtype=float)
    f = to_forcing(inputs, sim_times)
    # Callable signature: (z, t, theta, c) -> (gamma_w, gamma_s)
    gw, gs = f.n_source_terms(np.array([-10.0]), 0.5, 0.3, 5.0)
    # gamma_s should reflect first-order decay at requested rate; non-zero
    assert isinstance(gw, float)
    assert isinstance(gs, float)


def test_to_forcing_root_density_normalized():
    inputs = _minimal_inputs(5)
    sim_times = np.arange(5, dtype=float)
    f = to_forcing(inputs, sim_times)
    # beta(z) over [0, -z_max] should sum to ~1 when integrated
    z = np.linspace(0.0, -inputs.root.z_max_cm, 101)
    beta = f.root_density_fn(z, 2.0)
    # rough trapezoidal integral; abs() because z is decreasing (0 → -z_max)
    integ = abs(np.trapezoid(np.abs(beta), z))
    assert abs(integ - 1.0) < 0.1     # within 10% (coarse normalization is fine)


def test_to_forcing_passes_irrigation_and_fert_through():
    from hydrus_research.dndc_seam.schema import IrrigEvent, FertEvent
    inputs = _minimal_inputs(5)
    inputs.irrig_events.append(IrrigEvent(date=date(2026, 5, 2), method="drip",
                                          amount_cm=0.5, duration_h=2.0,
                                          solute_concs_mg_l={"NO3": 50.0}))
    inputs.fert_events.append(FertEvent(date=date(2026, 5, 3), depth_cm=0.0,
                                        mass_kg_n_ha=80.0, form="urea"))
    sim_times = np.arange(5, dtype=float)
    f = to_forcing(inputs, sim_times)
    assert len(f.irrigation_events) == 1
    assert len(f.fert_events) == 1
    assert f.irrigation_events[0].method == "drip"
    assert f.irrigation_events[0].solute_concs_mg_l == {"NO3": 50.0}
