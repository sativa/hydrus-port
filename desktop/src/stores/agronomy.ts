import { defineStore } from "pinia";
import {
  agronomy, type Crop, type Soil, type WeatherMeta,
  type AgronomyResult,
} from "../api";

type Irrig = { date: string; depth_mm: number };
type Fert  = { date: string; kg_n_ha: number; conc_mg_l?: number | null };

export const useAgronomyStore = defineStore("agronomy", {
  state: () => ({
    crops:    [] as Crop[],
    soils:    [] as Soil[],
    weather:  [] as WeatherMeta[],
    libsLoaded: false,

    cropId:    "maize",
    soilId:    "loam",
    weatherId: "n_china_avg",
    horizonDays: 150,
    startYear: 2026,

    irrig:     [] as Irrig[],
    fert:      [] as Fert[],

    result:    null as AgronomyResult | null,
    running:   false,
    error:     null as string | null,
  }),

  actions: {
    async loadLibs() {
      if (this.libsLoaded) return;
      [this.crops, this.soils, this.weather] = await Promise.all([
        agronomy.listCrops(), agronomy.listSoils(), agronomy.listWeather(),
      ]);
      this.libsLoaded = true;
    },
    addIrrig() { this.irrig.push({ date: this._defaultDate(), depth_mm: 20 }); },
    addFert()  { this.fert .push({ date: this._defaultDate(), kg_n_ha: 60 }); },
    removeIrrig(i: number) { this.irrig.splice(i, 1); },
    removeFert (i: number) { this.fert .splice(i, 1); },
    _defaultDate(): string {
      const d = new Date(this.startYear, 4, 15);  // May 15
      return d.toISOString().slice(0, 10);
    },
    async run() {
      this.running = true; this.error = null;
      try {
        this.result = await agronomy.run({
          crop_id: this.cropId, soil_id: this.soilId,
          weather_id: this.weatherId,
          horizon_days: this.horizonDays, start_year: this.startYear,
          irrigation: this.irrig, fertilizer: this.fert,
        });
      } catch (e: any) {
        this.error = String(e.message || e);
      } finally {
        this.running = false;
      }
    },
  },
});
