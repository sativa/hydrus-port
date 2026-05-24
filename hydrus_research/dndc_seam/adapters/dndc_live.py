"""DndcLiveAdapter — B2 placeholder. The real implementation calls the
user's Python DNDC each day to produce a DndcSeamInputs slice."""
from __future__ import annotations
from datetime import date

from . import DndcSeamAdapter
from ..schema import DndcSeamInputs


class DndcLiveAdapter(DndcSeamAdapter):
    """B2 stub. Will call self.dndc.export_hydrus_forcing(day_range)."""

    def __init__(self, dndc_session):
        self.dndc = dndc_session

    def produce(self, scenario_id: str, day_range: tuple[date, date]) -> DndcSeamInputs:
        raise NotImplementedError(
            "DndcLiveAdapter is a B2-placeholder; implement when wiring the "
            "real Python DNDC. See spec §3.2."
        )
