"""DNDC ↔ HYDRUS data seam.

Today: filled by manual GUI form / CSV / JSON file.
Tomorrow (B2): filled by `DndcLiveAdapter` calling the user's Python DNDC.

See DOCS/superpowers/specs/2026-05-24-hydrus-research-platform-design.md §3.

Lazy imports: sub-models are created incrementally across M1.2–M1.5;
DndcSeamInputs and to_forcing are created in M1.5 and M1.7.
"""

__all__ = ["DndcSeamInputs", "to_forcing"]


def __getattr__(name: str):
    if name == "DndcSeamInputs":
        from .schema import DndcSeamInputs
        return DndcSeamInputs
    elif name == "to_forcing":
        from .to_forcing import to_forcing
        return to_forcing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
