"""Weather typical-year CSV loader."""
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path


_DATA_DIR = Path(__file__).parent / "data" / "weather"

_NAMES_ZH = {
    "n_china_avg":    "华北平水年",
    "n_china_wet":    "华北丰水年",
    "n_china_dry":    "华北枯水年",
    "c_china_meiyu":  "华中梅雨",
    "s_china_double": "华南双季",
    "nw_china_irrig": "西北灌溉年",
}


def load_weather_meta() -> list[dict]:
    out = []
    for p in sorted(_DATA_DIR.glob("*.csv")):
        wid = p.stem
        out.append({"id": wid, "name_zh": _NAMES_ZH.get(wid, wid)})
    return out


@lru_cache(maxsize=8)
def _load_weather_series_cached(
    weather_id: str,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    p = _DATA_DIR / f"{weather_id}.csv"
    if p.resolve().parent != _DATA_DIR.resolve():
        raise KeyError(f"unknown weather id: {weather_id}")
    if not p.exists():
        raise KeyError(f"unknown weather id: {weather_id}")
    doy, P, PET = [], [], []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            doy.append(int(row["doy"]))
            P.append(float(row["P_mm"]))
            PET.append(float(row["PET_mm"]))
    return tuple(doy), tuple(P), tuple(PET)


def load_weather_series(weather_id: str) -> dict[str, list[float]]:
    doy, P, PET = _load_weather_series_cached(weather_id)
    return {"doy": list(doy), "P_mm": list(P), "PET_mm": list(PET)}
