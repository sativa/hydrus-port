"""ManualFileAdapter — load a DndcSeamInputs from a JSON or YAML file on disk."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import yaml

from . import DndcSeamAdapter
from ..schema import DndcSeamInputs


class ManualFileAdapter(DndcSeamAdapter):
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def produce(self, scenario_id: str, day_range: tuple[date, date]) -> DndcSeamInputs:
        text = self.path.read_text()
        if self.path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
            return DndcSeamInputs.model_validate(data)
        return DndcSeamInputs.model_validate_json(text)
