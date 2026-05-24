"""CsvNetcdfAdapter — bulk-research adapter. Takes a base DndcSeamInputs
(or base-preset file) plus optional CSV/NetCDF overrides for the
atmospheric series and event lists."""
from __future__ import annotations
import csv
from datetime import date, datetime
from pathlib import Path

from . import DndcSeamAdapter
from ..schema import DndcSeamInputs, AtmDaily, FertEvent, IrrigEvent


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


class CsvNetcdfAdapter(DndcSeamAdapter):
    def __init__(self,
                 *,
                 atm_csv: Path | str | None = None,
                 fert_csv: Path | str | None = None,
                 irrig_csv: Path | str | None = None,
                 base_inputs: DndcSeamInputs | None = None,
                 base_preset_path: Path | str | None = None):
        if base_inputs is None and base_preset_path is None:
            raise ValueError("must provide base_inputs or base_preset_path")
        if base_inputs is None:
            from .manual_file import ManualFileAdapter
            base_inputs = ManualFileAdapter(base_preset_path).produce("base",
                                                                      (date.today(), date.today()))
        self.base_inputs = base_inputs
        self.atm_csv = Path(atm_csv) if atm_csv else None
        self.fert_csv = Path(fert_csv) if fert_csv else None
        self.irrig_csv = Path(irrig_csv) if irrig_csv else None

    def _load_atm(self) -> AtmDaily:
        rows = list(csv.DictReader(self.atm_csv.open()))
        dates = [_parse_date(r["date"]) for r in rows]
        precip = [float(r["precip_cm"]) for r in rows]
        pet = [float(r["pet_cm"]) for r in rows] if rows and "pet_cm" in rows[0] else None
        return AtmDaily(dates=dates, precip_cm=precip, pet_cm=pet)

    def _load_fert(self) -> list[FertEvent]:
        rows = list(csv.DictReader(self.fert_csv.open()))
        return [FertEvent(date=_parse_date(r["date"]),
                          depth_cm=float(r.get("depth_cm", 0)),
                          mass_kg_n_ha=float(r["mass_kg_n_ha"]),
                          form=r["form"]) for r in rows]

    def _load_irrig(self) -> list[IrrigEvent]:
        rows = list(csv.DictReader(self.irrig_csv.open()))
        return [IrrigEvent(date=_parse_date(r["date"]),
                           method=r["method"],
                           amount_cm=float(r["amount_cm"]),
                           duration_h=float(r["duration_h"])) for r in rows]

    def produce(self, scenario_id: str, day_range: tuple[date, date]) -> DndcSeamInputs:
        di = self.base_inputs.model_copy(deep=True)
        if self.atm_csv:
            di = di.model_copy(update={"atm": self._load_atm()})
        if self.fert_csv:
            di = di.model_copy(update={"fert_events": self._load_fert()})
        if self.irrig_csv:
            di = di.model_copy(update={"irrig_events": self._load_irrig()})
        return di
