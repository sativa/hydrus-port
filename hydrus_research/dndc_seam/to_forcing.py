"""Convert the JSON-serializable DndcSeamInputs schema into a runtime Forcing
object (with concrete callables for root_density_fn and n_source_terms).

This is the SINGLE seam that connects the schema (Pydantic, REST-friendly)
to the engine's Forcing type (with closures, not serializable). Mode flags
in the schema (`density_profile=...`, `mode=constant_rates`, etc.) are
materialized into closures here."""
from __future__ import annotations
import numpy as np

from .schema import DndcSeamInputs, RootGrowth, NTransformation
from ..simulator.base import Forcing, Event


# ------------------------------------------------------------------- root growth
def _root_depth_curve(root: RootGrowth, sim_times_days: np.ndarray) -> np.ndarray:
    z_max = root.z_max_cm
    if root.growth_curve == "linear":
        d = root.days_to_zmax or float("inf")
        return np.minimum(z_max * sim_times_days / d, z_max)
    if root.growth_curve == "logistic":
        d = root.days_to_zmax or float("inf")
        # Standard logistic with inflection at d/2, asymptote z_max
        k = 8.0 / d                              # slope chosen so curve is ~99% at t=d
        return z_max / (1.0 + np.exp(-k * (sim_times_days - d / 2.0)))
    if root.growth_curve == "table":
        # `root.table` is list[(date, depth)] — convert to (days-since-first-entry, depth)
        ts = np.array([(t - root.table[0][0]).days for t, _ in root.table], dtype=float)
        zs = np.array([z for _, z in root.table], dtype=float)
        return np.clip(np.interp(sim_times_days, ts, zs), 0.0, z_max)
    raise ValueError(f"unknown growth_curve {root.growth_curve!r}")


def _root_density_fn(root: RootGrowth):
    z_max = root.z_max_cm
    param = root.density_param
    profile = root.density_profile

    def beta(z: np.ndarray, t: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        # Normalize to [0, 1] over [0, z_max] (z is negative going down)
        z_abs = np.clip(-z, 0.0, z_max)
        if profile == "uniform":
            b = np.where(z_abs <= z_max, 1.0 / z_max, 0.0)
        elif profile == "linear_decline":
            b = np.where(z_abs <= z_max, 2.0 * (1.0 - z_abs / z_max) / z_max, 0.0)
        elif profile == "exponential":
            k = param if param is not None else 3.0 / z_max
            b = np.where(z_abs <= z_max, k * np.exp(-k * z_abs), 0.0)
        elif profile == "raats":
            # Raats (1974) two-parameter; simplified
            k = param if param is not None else 5.0 / z_max
            b = np.where(z_abs <= z_max, k * np.exp(-k * z_abs), 0.0)
        else:
            raise ValueError(f"unknown density_profile {profile!r}")
        # Normalize so integral equals 1 (cm^-1)
        # Trapezoidal normalization across the requested grid;
        # use abs() because z may be decreasing (negative-down convention).
        if b.sum() > 0 and z.size > 1:
            denom = abs(np.trapezoid(np.abs(b), z))
            if denom > 0:
                b = b / denom
        return b

    return beta


# ------------------------------------------------------------------- N source terms
def _n_source_terms(n: NTransformation):
    """Returns callable (z, t, theta, c) -> (gamma_w, gamma_s).
    gamma_w: water source/sink term (1/T); gamma_s: solute reaction (1/T)."""
    if n.mode == "constant_rates":
        # First-order decay; gamma_s = -(k_nit + k_den + k_vol) * c
        ks = (n.k_nitrification_d or 0.0) + (n.k_denitrification_d or 0.0) \
             + (n.k_volatilization_d or 0.0)

        def fn(z, t, theta, c):
            return (0.0, -ks)     # interpreter: rate per day, applied per node
        return fn
    if n.mode == "external_callable":
        # Lazy import only when actually invoked — this is the B2 hook
        ref = n.callable_ref

        def fn(z, t, theta, c):
            mod_name, _, attr = ref.partition(":")
            import importlib
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr)(z, t, theta, c)
        return fn
    if n.mode == "lookup_table":
        # Stub: open the NetCDF and interpolate. Not implemented until M2.
        raise NotImplementedError("lookup_table mode lands in M2.x")
    raise ValueError(f"unknown NTransformation.mode {n.mode!r}")


# ------------------------------------------------------------------- atmospheric arrays
def _atm_arrays(inputs: DndcSeamInputs, sim_times: np.ndarray):
    """Interpolate daily atm series onto sim_times (which may be sub-daily)."""
    n_days = len(inputs.atm.dates)
    day_idx = np.arange(n_days, dtype=float)

    precip = np.interp(sim_times, day_idx, inputs.atm.precip_cm)
    if inputs.atm.pet_cm is not None:
        pet = np.interp(sim_times, day_idx, inputs.atm.pet_cm)
    else:
        pet = np.zeros_like(sim_times)            # caller computes from weather elsewhere
    t_air = None
    if inputs.atm.t_min_c is not None and inputs.atm.t_max_c is not None:
        t_air = 0.5 * (np.interp(sim_times, day_idx, inputs.atm.t_min_c)
                       + np.interp(sim_times, day_idx, inputs.atm.t_max_c))
    return precip, pet, t_air


def _lai_array(inputs: DndcSeamInputs, sim_times: np.ndarray) -> np.ndarray:
    if inputs.et.lai is not None:
        n_days = len(inputs.atm.dates)
        return np.interp(sim_times, np.arange(n_days, dtype=float), inputs.et.lai)
    if inputs.et.kcb is not None:
        n_days = len(inputs.atm.dates)
        return np.interp(sim_times, np.arange(n_days, dtype=float), inputs.et.kcb)
    return np.zeros_like(sim_times)


# ------------------------------------------------------------------- events conversion
def _convert_events(events_in, day0):
    """Convert schema events (with `date`) to Forcing Events (with `time_day` float)."""
    out = []
    for e in events_in:
        days = (e.date - day0).days
        if hasattr(e, "method"):
            out.append(Event(
                time_day=float(days), depth_cm=0.0, amount=e.amount_cm,
                method=e.method, solute_concs_mg_l=dict(e.solute_concs_mg_l),
            ))
        else:        # FertEvent
            out.append(Event(
                time_day=float(days), depth_cm=e.depth_cm, amount=e.mass_kg_n_ha,
                method="fert", solute_concs_mg_l={}, form=e.form,
            ))
    return out


# ------------------------------------------------------------------- entry point
def to_forcing(inputs: DndcSeamInputs, sim_times_days: np.ndarray) -> Forcing:
    """Build a runtime Forcing from a DndcSeamInputs schema instance.

    `sim_times_days` is the time axis of the simulation (in days since the
    start of `inputs.atm.dates[0]`). Daily schema series are interpolated
    onto this axis. The returned Forcing has live callables for
    root_density_fn and n_source_terms — not serializable, but ready for
    Simulator.run.
    """
    sim_times_days = np.asarray(sim_times_days, dtype=float)
    day0 = inputs.atm.dates[0]
    precip, pet, t_air = _atm_arrays(inputs, sim_times_days)
    return Forcing(
        times_days=sim_times_days,
        precip_cm_per_day=precip,
        pet_cm_per_day=pet,
        lai=_lai_array(inputs, sim_times_days),
        root_depth_cm=_root_depth_curve(inputs.root, sim_times_days),
        root_density_fn=_root_density_fn(inputs.root),
        irrigation_events=_convert_events(inputs.irrig_events, day0),
        fert_events=_convert_events(inputs.fert_events, day0),
        n_source_terms=_n_source_terms(inputs.n_transform),
        air_temp_c=t_air,
    )
