"""hydrus_research.agronomy — workflow-driven decision API."""
from .types import (
    AgronomyRequest, AgronomyResult,
    IrrigEvent, FertEvent,
    WaterBalance, NBudget, EventTick,
)
from .scenario_builder import build_scenario, event_to_t_day
from .runner import run_agronomy

__all__ = [
    "AgronomyRequest", "AgronomyResult",
    "IrrigEvent", "FertEvent",
    "WaterBalance", "NBudget", "EventTick",
    "build_scenario", "event_to_t_day",
    "run_agronomy",
]
