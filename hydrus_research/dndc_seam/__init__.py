"""DNDC ↔ HYDRUS data seam.

Today: filled by manual GUI form / CSV / JSON file.
Tomorrow (B2): filled by `DndcLiveAdapter` calling the user's Python DNDC.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §3.
"""
from .schema import DndcSeamInputs
from .to_forcing import to_forcing

__all__ = ["DndcSeamInputs", "to_forcing"]
