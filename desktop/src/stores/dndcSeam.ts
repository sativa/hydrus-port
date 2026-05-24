// Reactive DndcSeamInputs store. Sub-section components two-way bind via
// `useDndcSeamStore()`. The full DndcSeamInputs object is sent to the
// backend `/research/dndc/validate` on change (debounced).
import { defineStore } from "pinia";

export interface AtmDaily {
  dates: string[];                 // ISO YYYY-MM-DD
  precip_cm: number[];
  pet_cm: number[] | null;
}

export interface EtPartition {
  mode: "lai_beer" | "explicit_split" | "kc_dual";
  lai?: number[];
  extinction_k?: number;
  explicit_e_frac?: number[];
  kcb?: number[];
}

export interface RootGrowth {
  z_max_cm: number;
  growth_curve: "linear" | "logistic" | "table";
  days_to_zmax?: number;
  density_profile: "uniform" | "linear_decline" | "exponential" | "raats";
  density_param?: number;
}

export interface FeddesParams {
  h1: number; h2: number; h3_high: number; h3_low: number; h4: number;
  pet_high_cm_d: number; pet_low_cm_d: number;
}

export interface FertEvent {
  date: string;
  depth_cm: number;
  mass_kg_n_ha: number;
  form: "NH4" | "NO3" | "urea" | "NH4NO3" | "compound";
  composition?: Record<string, number>;
}

export interface IrrigEvent {
  date: string;
  method: "flood" | "sprinkler" | "drip" | "subsurface";
  amount_cm: number;
  duration_h: number;
  solute_concs_mg_l: Record<string, number>;
  drip_emitter_xyz?: [number, number, number];
}

export interface NTransformation {
  mode: "constant_rates" | "external_callable" | "lookup_table";
  k_mineralization_d?: number;
  k_nitrification_d?: number;
  k_denitrification_d?: number;
  k_volatilization_d?: number;
  callable_ref?: string;
}

export interface PlantNUptake {
  mode: "passive_with_water" | "michaelis_menten" | "demand_driven" | "external";
  km_mg_l?: number;
  vmax_mg_per_day_per_root_cm?: number;
  daily_demand_kg_n_ha?: number[];
  callable_ref?: string;
}

export interface StateExchange {
  z_grid_cm: number[];
  initial_theta?: number[];
  initial_h?: number[];
  initial_c?: Record<string, number[]>;
  initial_t?: number[];
  writeback_daily: boolean;
  writeback_path?: string;
}

export interface DndcSeamInputs {
  atm: AtmDaily;
  et: EtPartition;
  root: RootGrowth;
  feddes: FeddesParams;
  fert_events: FertEvent[];
  irrig_events: IrrigEvent[];
  n_transform: NTransformation;
  plant_n_uptake: PlantNUptake;
  state: StateExchange;
  soil_temp: { enabled: boolean; surface_t_daily_c?: number[] };
  residue: { mulch_fraction: number; residue_kg_ha: number; e_reduction_factor: number };
}

function blank(): DndcSeamInputs {
  return {
    atm: { dates: [], precip_cm: [], pet_cm: null },
    et: { mode: "lai_beer", lai: [], extinction_k: 0.6 },
    root: { z_max_cm: 50, growth_curve: "logistic", days_to_zmax: 30,
            density_profile: "linear_decline" },
    feddes: { h1: -15, h2: -30, h3_high: -325, h3_low: -600, h4: -8000,
              pet_high_cm_d: 0.5, pet_low_cm_d: 0.1 },
    fert_events: [],
    irrig_events: [],
    n_transform: { mode: "constant_rates", k_nitrification_d: 0.1 },
    plant_n_uptake: { mode: "passive_with_water" },
    state: { z_grid_cm: [0.0], writeback_daily: false },
    soil_temp: { enabled: false },
    residue: { mulch_fraction: 0.0, residue_kg_ha: 0.0, e_reduction_factor: 1.0 },
  };
}

export const useDndcSeamStore = defineStore("dndcSeam", {
  state: () => ({
    inputs: blank() as DndcSeamInputs,
    serverWarnings: [] as string[],
    serverError: null as string | null,
  }),
  actions: {
    reset() { this.inputs = blank(); this.serverWarnings = []; this.serverError = null; },
  },
});
