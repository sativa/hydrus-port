"""Canonical scenario schema.

This is the **single source of truth** for simulator inputs.
HYDRUS-1D's Selector.in+Profile.dat, SWMS_2D's SELECTOR.IN+GRID.IN,
and the 3D scikit-fem case format are all serialisations of this
same physical model — different dimensions, different ASCII
conventions, same physics.

The GUI parameter editor edits *this* structure. Adapters in
`hydrus_port/adapters/` convert to/from each legacy ASCII format.

Layout
------

    Scenario
    ├── meta        (heading, author, description, source)
    ├── units       (length, time, mass)
    ├── solver      (Picard iters, tolerances, physics flags)
    ├── materials   list[HydraulicMaterial]   van Genuchten-Mualem per material
    ├── time        (dt control, t_init/max, print_times)
    ├── root_uptake?    Feddes water-stress response
    ├── solute?         ADE + reactions
    ├── geometry    Geometry1D | Geometry2D | Geometry3D
    └── atmospheric?    time-variable BC table (Atmosph.in)

Geometry holds nodes/elements + per-node initial-head, mat-num,
boundary-code. Materials are referenced by index, not duplicated
into every node — define-then-assign.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Literal, Union
import json
from pathlib import Path


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------- meta

@dataclass
class ScenarioMeta:
    name: str = ""
    description: str = ""
    source: str = ""       # e.g. "imported from SWMS_2D EX1"


@dataclass
class Units:
    length: str = "cm"     # cm, m
    time:   str = "day"    # sec, min, hour, day
    mass:   str = "-"      # g, kg, -


# ---------------------------------------------------------------- solver

@dataclass
class Solver:
    """Picard control + physics flags (semantic, simulator-agnostic)."""
    # Geometry / coordinate system
    geometry_kind: Literal["horizontal", "axisymmetric", "vertical"] = "vertical"

    # Iteration control (shared by all three simulators)
    max_picard: int = 20
    tol_theta: float = 0.001
    tol_h: float = 1.0

    # Physics modules
    water_flow: bool = True
    solute_transport: bool = False
    heat_transport: bool = False
    root_uptake: bool = False

    # BC modes
    atmospheric_bc: bool = False
    free_drainage: bool = False
    seepage_face: bool = False
    subsurface_drain: bool = False
    gw_level: bool = False         # qGWLF in SWMS, similar in HYDRUS

    # IO
    short_output: bool = False
    flux_output: bool = True
    check_output: bool = False

    # Hysteresis / equilibrium options
    hysteresis: bool = False
    equilibrium: bool = True


# ---------------------------------------------------------------- materials

@dataclass
class HydraulicMaterial:
    """Van Genuchten-Mualem soil hydraulic parameters.

    The core 5 (`theta_r`, `theta_s`, `alpha`, `n`, `Ks`) are required;
    `Kk`/`theta_k` and `theta_a`/`theta_m` are the Vogel-Cislerova /
    SWMS_2D modifications and default to the corresponding "vanilla"
    values so a vanilla VG material round-trips cleanly through both
    schemas. `l` is Mualem's tortuosity (HYDRUS-1D has it, SWMS hard-
    codes 0.5)."""
    theta_r: float
    theta_s: float
    alpha: float           # 1/length
    n: float               # >1
    Ks: float              # length/time
    l: float = 0.5         # Mualem pore-connectivity, HYDRUS-1D exposes it
    # Vogel-Cislerova extensions (default to the simple VG case)
    theta_a: float | None = None      # default = theta_r
    theta_m: float | None = None      # default = theta_s
    theta_k: float | None = None      # default = theta_s
    Kk: float | None = None           # default = Ks

    def vc_theta_a(self) -> float: return self.theta_a if self.theta_a is not None else self.theta_r
    def vc_theta_m(self) -> float: return self.theta_m if self.theta_m is not None else self.theta_s
    def vc_theta_k(self) -> float: return self.theta_k if self.theta_k is not None else self.theta_s
    def vc_Kk(self) -> float:      return self.Kk      if self.Kk      is not None else self.Ks


# ---------------------------------------------------------------- time

@dataclass
class TimeControl:
    dt: float = 0.001
    dt_min: float = 1e-5
    dt_max: float = 1.0
    dt_mul: float = 1.3
    dt_mul2: float = 0.7
    it_min: int = 3
    it_max: int = 7
    t_init: float = 0.0
    t_max: float = 1.0
    print_times: list[float] = field(default_factory=list)


# ---------------------------------------------------------------- root uptake (Feddes)

@dataclass
class FeddesRootUptake:
    """Standard Feddes (1978) 4-piecewise stress-response curve."""
    p0: float = -10.0        # anaerobic
    p2H: float = -200.0      # optimum lower bound (high transp)
    p2L: float = -800.0      # optimum lower bound (low transp)
    p3: float = -8000.0      # wilting
    r2H: float = 0.5
    r2L: float = 0.1
    p_optm: list[float] = field(default_factory=list)   # per material


# ---------------------------------------------------------------- solute (ADE + reactions)

@dataclass
class SoluteTransport:
    epsi: float = 0.5
    upwind: bool = False
    artificial_dispersion: bool = False
    pe_crit: float = 0.0
    # Per-material chemical parameters; one row per material, fixed length
    # following SWMS_2D's CHEM block: Bulk.d / Difus / Disper / Adsorp /
    # SinkL1 / SinkS1 / SinkL0 / SinkS0 / Frac
    chem_params: list[list[float]] = field(default_factory=list)
    # Boundary
    kod_cb: list[int] = field(default_factory=list)        # one per boundary node
    c_bound: list[float] = field(default_factory=list)     # c values per boundary segment
    t_pulse: float = 0.0


# ---------------------------------------------------------------- geometry (dim-specific)

@dataclass
class Geometry1D:
    kind: Literal["1d"] = "1d"
    z: list[float] = field(default_factory=list)         # depths (top first, decreasing)
    initial_h: list[float] = field(default_factory=list) # per node
    mat_num: list[int] = field(default_factory=list)     # 1-based
    layer: list[int] = field(default_factory=list)       # 1-based
    beta: list[float] = field(default_factory=list)      # root distribution
    # Geometric scaling factors (HYDRUS-1D Axz/Bxz/Dxz)
    axz: list[float] = field(default_factory=list)
    bxz: list[float] = field(default_factory=list)
    dxz: list[float] = field(default_factory=list)


@dataclass
class Geometry2D:
    kind: Literal["2d"] = "2d"
    # Nodes
    x: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)         # vertical (SWMS calls it "y")
    initial_h: list[float] = field(default_factory=list)
    mat_num: list[int] = field(default_factory=list)
    kode: list[int] = field(default_factory=list)        # BC code per node
    # Per-node anisotropy scaling (Axz/Bxz/Dxz family)
    axz: list[float] = field(default_factory=list)
    bxz: list[float] = field(default_factory=list)
    dxz: list[float] = field(default_factory=list)
    # Per-node solute concentration + flux + root distribution
    conc: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    # Connectivity: 0-based [n1, n2, n3, n4]; for tris n4 == n3
    elements: list[list[int]] = field(default_factory=list)
    # Per-element anisotropy tensor (already rotated to global x/z)
    con_axx: list[float] = field(default_factory=list)
    con_azz: list[float] = field(default_factory=list)
    con_axz: list[float] = field(default_factory=list)
    layer_per_element: list[int] = field(default_factory=list)
    # Boundary
    boundary_nodes: list[int] = field(default_factory=list)
    boundary_width: list[float] = field(default_factory=list)
    boundary_total_length: float = 0.0


@dataclass
class Geometry3D:
    kind: Literal["3d"] = "3d"
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    initial_h: list[float] = field(default_factory=list)
    mat_num: list[int] = field(default_factory=list)
    cells: list[list[int]] = field(default_factory=list)
    cell_kind: Literal["tet", "hex"] = "tet"
    boundary_nodes: list[int] = field(default_factory=list)


Geometry = Union[Geometry1D, Geometry2D, Geometry3D]


# ---------------------------------------------------------------- atmospheric BC time series

@dataclass
class AtmosphericRow:
    t: float
    precip: float = 0.0
    evap: float = 0.0
    h_critA: float = -1e30   # critical pressure threshold for atm BC
    rRoot: float = 0.0
    rB: float = 0.0          # bottom water flux
    hB: float = 0.0          # bottom prescribed head
    ht: float = 0.0          # top prescribed head
    tTop: float = 0.0
    tBot: float = 0.0
    Ampl: float = 0.0


@dataclass
class AtmosphericBC:
    sink_flag: bool = False
    qgwlf: bool = False
    rows: list[AtmosphericRow] = field(default_factory=list)


# ---------------------------------------------------------------- seepage / drain

@dataclass
class SeepageFace:
    """SWMS_2D BLOCK E."""
    nodes_per_face: list[int] = field(default_factory=list)   # NSP[i]
    face_node_lists: list[list[int]] = field(default_factory=list)


@dataclass
class SubsurfaceDrain:
    correction: float = 1.0
    nodes_per_drain: list[int] = field(default_factory=list)
    drain_nodes: list[int] = field(default_factory=list)
    elements_per_drain: list[int] = field(default_factory=list)
    effective_dimensions: list[list[float]] = field(default_factory=list)
    drain_elements: list[list[int]] = field(default_factory=list)


# ---------------------------------------------------------------- the umbrella

@dataclass
class Scenario:
    """Canonical, simulator-agnostic scenario."""
    schema_version: str = SCHEMA_VERSION
    meta: ScenarioMeta = field(default_factory=ScenarioMeta)
    units: Units = field(default_factory=Units)
    solver: Solver = field(default_factory=Solver)
    materials: list[HydraulicMaterial] = field(default_factory=list)
    time: TimeControl = field(default_factory=TimeControl)
    root_uptake: FeddesRootUptake | None = None
    solute: SoluteTransport | None = None
    # Default geometry to 1D so a fresh Scenario() doesn't error
    geometry: Geometry = field(default_factory=Geometry1D)
    atmospheric: AtmosphericBC | None = None
    seepage: SeepageFace | None = None
    drain: SubsurfaceDrain | None = None
    # Adapter-specific carryover that we don't model but want to
    # round-trip unchanged (e.g. heading text, extra solver flags
    # unique to one simulator). Adapters set this.
    legacy_extras: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> Literal["1d", "2d", "3d"]:
        return self.geometry.kind

    # ---- (de)serialisation -------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_dict(self) -> dict:
        d = _asdict_lossless(self)
        # Carry the geometry's discriminant as the JSON `kind` key
        # (asdict already includes it, but make sure it's a plain str).
        d["geometry"]["kind"] = self.geometry.kind
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return _scenario_from_dict(d)

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_path(cls, path: str | Path) -> "Scenario":
        return cls.from_json(Path(path).read_text())

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


# ---------------------------------------------------------------- helpers

def _asdict_lossless(obj: Any) -> Any:
    """Like dataclasses.asdict, but recursive over Optional[dataclass]."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _asdict_lossless(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict_lossless(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict_lossless(v) for k, v in obj.items()}
    return obj


def _scenario_from_dict(d: dict) -> "Scenario":
    g = d.get("geometry", {})
    kind = g.get("kind", "1d")
    if kind == "1d":
        geom: Geometry = Geometry1D(**{k: v for k, v in g.items() if k != "kind"})
    elif kind == "2d":
        geom = Geometry2D(**{k: v for k, v in g.items() if k != "kind"})
    elif kind == "3d":
        geom = Geometry3D(**{k: v for k, v in g.items() if k != "kind"})
    else:
        raise ValueError(f"unknown geometry kind: {kind!r}")

    def _opt(cls, key):
        v = d.get(key)
        return cls(**v) if v else None

    return Scenario(
        schema_version=d.get("schema_version", SCHEMA_VERSION),
        meta=ScenarioMeta(**d.get("meta", {})),
        units=Units(**d.get("units", {})),
        solver=Solver(**d.get("solver", {})),
        materials=[HydraulicMaterial(**m) for m in d.get("materials", [])],
        time=TimeControl(**d.get("time", {})),
        root_uptake=_opt(FeddesRootUptake, "root_uptake"),
        solute=_opt(SoluteTransport, "solute"),
        geometry=geom,
        atmospheric=(AtmosphericBC(
            sink_flag=d["atmospheric"].get("sink_flag", False),
            qgwlf=d["atmospheric"].get("qgwlf", False),
            rows=[AtmosphericRow(**r) for r in d["atmospheric"].get("rows", [])],
        ) if d.get("atmospheric") else None),
        seepage=_opt(SeepageFace, "seepage"),
        drain=_opt(SubsurfaceDrain, "drain"),
        legacy_extras=d.get("legacy_extras", {}),
    )
