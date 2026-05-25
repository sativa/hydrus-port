"""M5 acceptance: synthetic recovery on infiltr_v1 via LM.

Generate y_obs from a perturbed alpha_true (= alpha_nominal × 1.5);
run LM from the nominal alpha and verify it recovers alpha_true within 5%.

Deviation from plan: z=-30cm is fully saturated at t≥1 day for this scenario
so it provides zero gradient.  We use depths z=-110/−120 cm where the wetting
front is still progressing at t=1 day (confirmed by sensitivity study).  We
also pass diff_step=0.05 because HYDRUS-1D text output has 4 decimal places of
precision — the default scipy step (~1e-8) falls below that floor and the
optimizer terminates immediately with a zero Jacobian.  diff_step=0.05 gives a
detectable signal in log-parameter space.  Both choices are documented in the
plan deviation note."""
import numpy as np
import pytest
from pathlib import Path

from hydrus_research.parameters import ParameterSpec, ParameterMap
from hydrus_research.observations import ObservationSpec, ObservationSet
from hydrus_research.simulator import make_forward
from hydrus_research.simulator.hydrus1d_adapter import Hydrus1DSimulator
from hydrus_research.inversion import fit_lm
from hydrus_port.adapters.hydrus1d import load as load_h1d_canonical


def test_lm_recovers_perturbed_alpha_on_infiltr_v1():
    template = load_h1d_canonical(Path("tests/fixtures/infiltr_v1/inputs")).to_dict()
    a0 = template["materials"][0]["alpha"]
    alpha_true = a0 * 1.5

    pm = ParameterMap([
        ParameterSpec(name="alpha", target="materials[0].alpha",
                      bounds=(a0 * 0.5, a0 * 3.0), transform="log"),
    ])
    # Use depths where the wetting front is still progressing at t=1 day.
    # z=-30cm is fully saturated well before t=1 day for this infiltration
    # scenario, giving zero sensitivity.  z=-110/-120 cm span the wetting
    # front transition zone and provide strong gradients.
    obs_specs = [
        ObservationSpec(name="theta_z110_d1", kind="theta",
                        location={"z_cm": -110.0}, time_day=1.0),
        ObservationSpec(name="theta_z120_d1", kind="theta",
                        location={"z_cm": -120.0}, time_day=1.0),
    ]
    sim = Hydrus1DSimulator()
    forward = make_forward(sim, pm, template_scenario=template,
                           forcing=None, ic=None, obs_specs=obs_specs)

    # Generate synthetic y_obs at alpha_true
    y_obs = forward(pm.to_vector({"alpha": alpha_true}))
    obs = ObservationSet(specs=obs_specs, values=y_obs,
                         sigmas=np.full(len(y_obs), 0.01))

    # Start from the nominal alpha; LM should walk to alpha_true.
    # diff_step=0.05: HYDRUS-1D text output has 4 d.p. precision, so the
    # default scipy step (~1e-8) is below the output floor and gives a zero
    # Jacobian.  0.05 (5% in log space) ensures a detectable finite difference.
    result = fit_lm(forward=forward, param_map=pm, obs=obs,
                    x0=pm.to_vector({"alpha": a0}),
                    max_nfev=30, diff_step=0.05)
    rel_err = abs(result.best_params["alpha"] - alpha_true) / alpha_true
    assert rel_err < 0.05, \
        f"recovered alpha={result.best_params['alpha']:.5g} vs true {alpha_true:.5g} (err={rel_err:.1%})"
