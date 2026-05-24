"""DndcSeamAdapter contract + concrete implementations.

Workflows (research modules in M2+) accept `DndcSeamAdapter`, never concrete
classes. That is the B2 plug-and-play guarantee."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date

from ..schema import DndcSeamInputs


class DndcSeamAdapter(ABC):
    """Anything that produces a DndcSeamInputs for a given scenario / day range."""

    @abstractmethod
    def produce(self, scenario_id: str,
                day_range: tuple[date, date]) -> DndcSeamInputs: ...


# Re-export concrete adapters at the package level
from .manual_file import ManualFileAdapter
from .dndc_live import DndcLiveAdapter

__all__ = ["DndcSeamAdapter", "ManualFileAdapter", "DndcLiveAdapter"]
