"""Unified `hydrus` CLI — dispatches to 1D / 2D / 3D solvers.

Usage
-----

    hydrus 1d <input_dir> [-o OUT]              HYDRUS-1D
    hydrus 2d <input_dir> [-o OUT] [...]        SWMS_2D
    hydrus 3d [<input_dir>] [-o OUT]            Richards 3D (scikit-fem)
    hydrus test [1d|2d|3d|all]                  Run smoke tests

Each subcommand defaults `-o` to `<input_dir>/out`. For 3D the input
positional is optional and defaults to running the built-in synthetic
box demo from tests/validate_richards3d.py.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


def _default_out(input_dir: Path | None) -> Path:
    if input_dir is None:
        return Path.cwd() / "hydrus_out"
    return input_dir / "out"


def _run_1d(args: argparse.Namespace) -> int:
    from hydrus1d.hydrus import run_simulation
    out = args.output_dir or _default_out(args.input_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_simulation(input_dir=str(args.input_dir), output_dir=str(out))
    if getattr(args, "csv", False):
        _emit_csv(out, args.input_dir)
    return 0


def _run_2d(args: argparse.Namespace) -> int:
    from swms2d.swms2d import SWMS2DSimulation
    out = args.output_dir or _default_out(args.input_dir)
    sim = SWMS2DSimulation(
        input_dir=args.input_dir,
        output_dir=out,
        use_anderson=args.anderson,
        anderson_m=args.anderson_m,
        use_banded=args.banded,
        write_vtk=args.vtk,
    )
    sim.run(verbose=not args.quiet)
    if getattr(args, "csv", False):
        _emit_csv(out, args.input_dir)
    return 0


def _emit_csv(out_dir: Path, input_dir: Path | None = None) -> None:
    """Run the CSV converter on out_dir; pass GRID.IN if available for
    SWMS_2D field exports (so x/z columns are populated)."""
    from .csv_export import convert_output_dir
    grid_path = None
    if input_dir is not None:
        for p in Path(input_dir).iterdir():
            if p.name.lower() == "grid.in":
                grid_path = p
                break
    written = convert_output_dir(out_dir, grid_path=grid_path)
    if written:
        print(f"wrote {len(written)} CSV file(s):", file=sys.stderr)
        for p in written:
            print(f"  {p}", file=sys.stderr)


def _run_3d(args: argparse.Namespace) -> int:
    # The current 3D entry point is the validation demo. When a real
    # mesh-driven scenario format is defined we can dispatch on that.
    if args.input_dir is None:
        from tests.validate_richards3d import main as r3d_demo
        rc = r3d_demo()
        if getattr(args, "csv", False):
            r3d_out = _repo_path("tests", "fixtures", "richards3d_box", "out")
            _emit_csv(r3d_out)
        return rc
    raise SystemExit(
        "3D mesh-driven inputs are not wired in yet. For now run "
        "`hydrus 3d` (no args) to execute the synthetic box demo."
    )


# ----------------------------------------------------------------------
# `hydrus test` — smoke-test each simulation path
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def _summarise_1d_outputs(out_dir: Path) -> dict:
    """Pull key numbers out of NOD_INF.OUT / Balance.out for reporting."""
    summary: dict = {"output_dir": str(out_dir), "files": []}
    if not out_dir.exists():
        return summary
    summary["files"] = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    bal = out_dir / "BALANCE.OUT"
    if bal.exists():
        # Last numeric line is usually the final time-step balance entry.
        lines = [l for l in bal.read_text().splitlines() if l.strip()]
        last = next((l for l in reversed(lines)
                     if l.strip().split()[0].lstrip("-").replace(".", "").isdigit()),
                    None)
        if last:
            summary["balance_last_line"] = last.strip()
    return summary


def _summarise_2d_outputs(out_dir: Path) -> dict:
    summary: dict = {"output_dir": str(out_dir), "files": []}
    if not out_dir.exists():
        return summary
    summary["files"] = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    bal = out_dir / "Balance.out"
    if bal.exists():
        # Pull the last "Volume" / "WatBalT" lines as a coarse mass-balance check
        lines = bal.read_text().splitlines()
        for key in ("Volume", "WatBalT", "WatBalR"):
            hits = [l for l in lines if key in l]
            if hits:
                summary[f"{key}_last"] = hits[-1].strip()
    return summary


def _test_1d(verbose: bool = True) -> tuple[bool, dict]:
    """Run HYDRUS-1D on soil_loam_infiltr."""
    input_dir = _repo_path("tests", "fixtures", "soil_loam_infiltr", "inputs")
    out = _repo_path("tests", "fixtures", "soil_loam_infiltr", "inputs", "out")
    if out.exists():
        for p in out.iterdir():
            p.unlink()
    if verbose:
        print(f"  input: {input_dir}")
    t0 = time.time()
    from hydrus1d.hydrus import run_simulation
    sim = run_simulation(input_dir=str(input_dir), output_dir=str(out))
    dt = time.time() - t0
    summary = _summarise_1d_outputs(out)
    final_t = getattr(getattr(sim, "state", None), "time", None)
    summary["wall_s"] = round(dt, 3)
    summary["sim_t"] = float(final_t.t) if final_t is not None else None
    expected = {"BALANCE.OUT", "NOD_INF.OUT", "T_LEVEL.OUT"}
    have = set(summary.get("files", []))
    ok = expected.issubset(have)
    return ok, summary


def _test_2d(verbose: bool = True) -> tuple[bool, dict]:
    """Run SWMS_2D EX1."""
    input_dir = _repo_path("tests", "fixtures", "EX1", "inputs")
    out = _repo_path("tests", "fixtures", "EX1", "inputs", "out")
    if out.exists():
        for p in out.iterdir():
            p.unlink()
    if verbose:
        print(f"  input: {input_dir}")
    t0 = time.time()
    from swms2d.swms2d import SWMS2DSimulation
    sim = SWMS2DSimulation(input_dir, out)
    sim.run(verbose=False)
    dt = time.time() - t0
    summary = _summarise_2d_outputs(out)
    summary["wall_s"] = round(dt, 3)
    expected = {"Balance.out", "h.out", "th.out", "Run_Inf.out"}
    have = set(summary.get("files", []))
    ok = expected.issubset(have)
    return ok, summary


def _test_3d(verbose: bool = True) -> tuple[bool, dict]:
    """Run the 3D Richards box validation (Tet/Hex × lumped/consistent)."""
    if verbose:
        print("  Tet/Hex × lumped/consistent cross-validation")
    t0 = time.time()
    from tests.validate_richards3d import run_case
    results: dict = {}
    import numpy as np
    for element in ("tetra", "hex"):
        for lump in (True, False):
            tag = f"{element}/{'lump' if lump else 'consistent'}"
            zs, hs, ths = run_case(element, lump)
            results[tag] = (zs, hs, ths)
    dt = time.time() - t0
    summary: dict = {"wall_s": round(dt, 3), "cases": {}}
    threshold = 20.0
    ok = True
    for element in ("tetra", "hex"):
        _, h_l, _ = results[f"{element}/lump"]
        _, h_c, _ = results[f"{element}/consistent"]
        d = float(np.max(np.abs(h_l - h_c)))
        summary["cases"][f"{element}_lump_vs_consistent_max_dh_cm"] = round(d, 3)
        if d > threshold:
            ok = False
    z_t, h_t, _ = results["tetra/lump"]
    z_h, h_h, _ = results["hex/lump"]
    h_h_at_t = np.interp(z_t, z_h, h_h)
    cross = float(np.max(np.abs(h_t - h_h_at_t)))
    summary["cases"]["tetra_vs_hex_lump_max_dh_cm"] = round(cross, 3)
    if cross > threshold:
        ok = False
    return ok, summary


def _print_summary(name: str, ok: bool, summary: dict) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {name}")
    for k, v in summary.items():
        if k == "files":
            print(f"  files ({len(v)}): {', '.join(v)}")
        elif k == "cases":
            for ck, cv in v.items():
                print(f"  {ck}: {cv}")
        else:
            print(f"  {k}: {v}")


def _test_roundtrip(verbose: bool = True) -> tuple[bool, dict]:
    """EX1 SELECTOR.IN round-trip: parse → write → parse → assert equal."""
    if verbose:
        print("  EX1 SELECTOR.IN parse → write → parse equivalence")
    from tests.test_round_trip_selector import round_trip
    in_dir = _repo_path("tests", "fixtures", "EX1", "inputs")
    ok, diffs = round_trip(in_dir)
    return ok, {"diffs": len(diffs), "first": diffs[:3]}


def _test_cases(verbose: bool = True) -> tuple[bool, dict]:
    """Run every canonical case in tests/cases/ via the unified adapter
    layer. Pass criterion: each case yields >=4 output files."""
    cases_dir = _repo_path("tests", "cases")
    if not cases_dir.exists():
        return False, {"error": "no tests/cases dir"}
    import tempfile
    from .schema import Scenario
    summary: dict = {"cases": {}}
    overall = True
    for case in sorted(cases_dir.glob("*.json")):
        sc = Scenario.from_path(case)
        ad = _adapter_for(sc.dimension)
        with tempfile.TemporaryDirectory(prefix="hydrus-case-test-") as td:
            ad.save(sc, td)
            try:
                if sc.dimension == "2d":
                    from swms2d.swms2d import SWMS2DSimulation
                    SWMS2DSimulation(td, td + "/out").run(verbose=False)
                elif sc.dimension == "1d":
                    from hydrus1d.hydrus import run_simulation
                    run_simulation(input_dir=td, output_dir=td + "/out")
                import os
                files = os.listdir(td + "/out")
                summary["cases"][case.stem] = (
                    f"{sc.dimension} {len(files)} files ✓"
                    if len(files) >= 4 else
                    f"{sc.dimension} {len(files)} files ✗"
                )
                if len(files) < 4:
                    overall = False
            except Exception as e:
                summary["cases"][case.stem] = f"FAILED: {e}"
                overall = False
    return overall, summary


def _run_test(args: argparse.Namespace) -> int:
    targets = (["1d", "2d", "3d", "roundtrip", "cases"]
               if args.target == "all" else [args.target])
    runners = {"1d": _test_1d, "2d": _test_2d, "3d": _test_3d,
               "roundtrip": _test_roundtrip, "cases": _test_cases}
    overall_ok = True
    results = []
    for t in targets:
        print(f"\n=== hydrus test {t} ===")
        try:
            ok, summary = runners[t]()
        except Exception as e:
            ok, summary = False, {"error": repr(e)}
        results.append((t, ok, summary))
        _print_summary(t, ok, summary)
        overall_ok = overall_ok and ok
    print("\n" + "=" * 40)
    print("OVERALL: " + ("PASS" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


# ----------------------------------------------------------------------
# `hydrus scenario {read,write}` — JSON bridge for the GUI editor
# ----------------------------------------------------------------------

def _scenario_to_dict(cfg, materials, time, extras) -> dict:
    import numpy as np
    def _np(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    return {
        "heading": extras.get("heading", ""),
        "units":   extras.get("units", []),
        "config": {
            "KAT": cfg.KAT, "MaxIt": cfg.MaxIt,
            "TolTh": cfg.TolTh, "TolH": cfg.TolH,
            "lWat": cfg.lWat, "lChem": cfg.lChem,
            "CheckF": cfg.CheckF, "ShortF": cfg.ShortF,
            "FluxF": cfg.FluxF, "AtmInF": cfg.AtmInF,
            "SeepF": cfg.SeepF, "FreeD": cfg.FreeD, "DrainF": cfg.DrainF,
        },
        "materials": [{
            "thr": m.thr, "ths": m.ths, "tha": m.tha, "thm": m.thm,
            "alpha": m.alpha, "n": m.n, "Ks": m.Ks,
            "Kk": m.Kk, "thk": m.thk,
        } for m in materials],
        "time": {
            "dt": time.dt, "dtMin": time.dtMin, "dtMaxW": time.dtMaxW,
            "dMul": time.dMul, "dMul2": time.dMul2,
        },
        "NLay": extras.get("NLay"),
        "hTab1": extras.get("hTab1"),
        "hTabN": extras.get("hTabN"),
        "NPar": extras.get("NPar"),
        "TPrint": list(_np(extras.get("TPrint", []))),
        "extras_raw": {
            k: _np(v) for k, v in extras.items()
            if k not in ("heading", "units", "TPrint",
                          "NLay", "hTab1", "hTabN", "NPar")
        },
    }


def _dict_to_scenario(d: dict):
    """Inverse of _scenario_to_dict — reconstruct the structures parse_selector returns."""
    import numpy as np
    from swms2d.dataclasses import SimulationConfig, SoilMaterial, TimeControl
    cfg = SimulationConfig()
    for k, v in d.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    mats = [SoilMaterial(**m) for m in d.get("materials", [])]
    t = TimeControl()
    for k, v in d.get("time", {}).items():
        if hasattr(t, k):
            setattr(t, k, v)
    extras = {
        "heading": d.get("heading", ""),
        "units":   d.get("units", []),
        "TPrint":  np.array(d.get("TPrint", []), np.float64),
        "NLay":    d.get("NLay"),
        "hTab1":   d.get("hTab1"),
        "hTabN":   d.get("hTabN"),
        "NPar":    d.get("NPar"),
    }
    # Re-hydrate anything else that came through raw
    raw = d.get("extras_raw", {})
    for k, v in raw.items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)) and k.startswith(("sink_", "drain_", "chem_")):
            extras[k] = np.array(v, np.float64) if "POptm" in k or "ChPar" in k else v
        else:
            extras[k] = v
    # Last derived: time.tMax from TPrint
    if len(extras["TPrint"]):
        t.tMax = float(extras["TPrint"][-1])
    return cfg, mats, t, extras


def _detect_dimension(input_dir: Path) -> str:
    """Sniff the input directory to decide which adapter to use."""
    # 3D canonical scenarios live as a single .json file
    if input_dir.is_file() and input_dir.suffix == ".json":
        return "3d"
    names_lower = {p.name.lower() for p in input_dir.iterdir()}
    if "scenario.json" in names_lower:        # 3D canonical
        return "3d"
    if "grid.in" in names_lower:              # SWMS_2D unique marker
        return "2d"
    if "profile.dat" in names_lower:          # HYDRUS-1D unique marker
        return "1d"
    if "selector.in" in names_lower:
        # ambiguous — peek at the file header to decide
        sel = next(p for p in input_dir.iterdir()
                   if p.name.lower() == "selector.in")
        head = sel.read_text(errors="ignore")[:80]
        return "1d" if head.startswith("Pcp_File_Version") else "2d"
    raise SystemExit(
        f"can't detect simulator format in {input_dir} "
        f"(need GRID.IN or Profile.dat or Selector.in or scenario.json)"
    )


def _adapter_for(dim: str):
    """Return the adapter module for the given dimension."""
    if dim in ("1d", "hydrus1d"):
        from .adapters import hydrus1d as a
        return a
    if dim in ("2d", "swms2d"):
        from .adapters import swms2d as a
        return a
    if dim in ("3d", "richards3d"):
        from .adapters import richards3d as a
        return a
    raise SystemExit(f"no adapter for dimension {dim!r}")


def _run_scenario(args: argparse.Namespace) -> int:
    """Unified scenario read/write — auto-detects 1d vs 2d source format,
    canonical JSON is the on-wire format in both directions."""
    import json
    in_dir = Path(args.input_dir).expanduser().resolve()
    if args.action == "read":
        dim = _detect_dimension(in_dir)
        scenario = _adapter_for(dim).load(in_dir)
        print(scenario.to_json())
        return 0
    elif args.action == "write":
        # Read JSON from stdin or --json file
        if args.json:
            payload = json.loads(Path(args.json).read_text())
        else:
            payload = json.loads(sys.stdin.read())
        from .schema import Scenario
        scenario = Scenario.from_dict(payload)
        out_dir = Path(args.out).expanduser().resolve() if args.out else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _adapter_for(scenario.dimension).save(scenario, out_dir)
        print(f"wrote {scenario.dimension} scenario → {out_dir}", file=sys.stderr)
        return 0
    raise SystemExit(f"unknown scenario action: {args.action}")


def _run_case(args: argparse.Namespace) -> int:
    """`hydrus run <case.json>` — runs a canonical scenario by writing
    it to a temp dir via the appropriate adapter and dispatching to
    the matching dimensional driver."""
    import json
    import tempfile
    from .schema import Scenario
    scen_path = Path(args.case).expanduser().resolve()
    scenario = Scenario.from_path(scen_path)
    print(f"running {scenario.dimension} case: {scenario.meta.name!r}",
          file=sys.stderr)
    out = args.output_dir or (scen_path.parent / f"{scen_path.stem}_out")
    Path(out).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hydrus-case-") as td:
        ad = _adapter_for(scenario.dimension)
        ad.save(scenario, td)
        # Some adapters (e.g. swms2d) need an existing GRID.IN; if so the
        # case writer should have already produced one or the user must
        # be running from a fixture that already has it.
        if scenario.dimension == "2d":
            from swms2d.swms2d import SWMS2DSimulation
            SWMS2DSimulation(td, str(out)).run(verbose=not args.quiet)
        elif scenario.dimension == "1d":
            from hydrus1d.hydrus import run_simulation
            run_simulation(input_dir=td, output_dir=str(out))
        elif scenario.dimension == "3d":
            from .adapters import richards3d as r3d
            r3d.run(scenario, out)
        else:
            raise SystemExit(f"unknown scenario dimension {scenario.dimension!r}")
    if getattr(args, "csv", False):
        # GRID.IN may live in the canonical Geometry2D; for SWMS_2D
        # we wrote a GRID.IN into the temp dir, but it's gone now —
        # the converter falls back gracefully without grid_path.
        _emit_csv(Path(out))
    print(f"output → {out}", file=sys.stderr)
    return 0


def _run_convert(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"not a directory: {out_dir}")
    grid_path = None
    if args.input_dir:
        for p in Path(args.input_dir).iterdir():
            if p.name.lower() == "grid.in":
                grid_path = p
                break
    from .csv_export import convert_output_dir
    written = convert_output_dir(out_dir, grid_path=grid_path)
    if not written:
        print("no recognised .out files in this directory", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} CSV file(s):", file=sys.stderr)
    for p in written:
        print(f"  {p}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ research subcommand group


def _cmd_dndc_list(args: argparse.Namespace) -> int:
    from hydrus_research.dndc_seam.presets import list_crop_presets, load_crop_preset

    for name in list_crop_presets():
        _, _, desc = load_crop_preset(name)
        print(f"  {name:12s} — {desc}")
    return 0


def _cmd_dndc_validate(args: argparse.Namespace) -> int:
    from hydrus_research.dndc_seam import DndcSeamInputs
    from pydantic import ValidationError

    try:
        DndcSeamInputs.model_validate_json(open(args.json_path).read())
    except ValidationError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print("ok — input is valid")
    return 0


def _cmd_dndc_to_forcing(args: argparse.Namespace) -> int:
    import numpy as _np
    from hydrus_research.dndc_seam import DndcSeamInputs, to_forcing

    di = DndcSeamInputs.model_validate_json(open(args.json_path).read())
    sim_times = _np.array([float(x) for x in args.sim_times.split(",")])
    f = to_forcing(di, sim_times)
    print(
        f"Forcing built: times[{f.times_days.shape}], "
        f"precip[{f.precip_cm_per_day.sum():.3f}cm total], "
        f"pet[{f.pet_cm_per_day.sum():.3f}cm total], "
        f"events: {len(f.irrigation_events)} irrig, {len(f.fert_events)} fert"
    )
    return 0


def _cmd_soil_ptf(args: argparse.Namespace) -> int:
    """Execute `hydrus research soil ptf`."""
    from hydrus_research.ptf import texture_to_vg
    parts = dict(p.split("=") for p in args.texture.split(","))
    r = texture_to_vg(
        sand_pct=float(parts["sand"]), silt_pct=float(parts["silt"]),
        clay_pct=float(parts["clay"]),
        bulk_density_g_cm3=args.bd, theta_33=args.theta_33,
        theta_1500=args.theta_1500, organic_matter_pct=args.om,
        method=args.method,
    )
    print(f"{'theta_r':12s} {'theta_s':12s} {'alpha':12s} {'n':12s} {'Ks':12s} method")
    print(f"{r.theta_r:<12.4f} {r.theta_s:<12.4f} {r.alpha:<12.4f} "
          f"{r.n:<12.4f} {r.Ks:<12.4f} {r.method}")
    return 0


def _build_research_subparser(sub: "argparse._SubParsersAction") -> None:
    """`hydrus research <subcmd>` — research-platform CLI surface.

    M1 adds: research dndc {list-presets, validate, to-forcing}
    M2 adds: research soil ptf
    """
    p_research = sub.add_parser(
        "research",
        help="research-platform tools (DNDC seam, PTF, ...)",
    )
    rsub = p_research.add_subparsers(dest="research_cmd", required=True,
                                     metavar="{dndc,soil}")

    # ----- dndc (M1) ---------------------------------------------------
    p_dndc = rsub.add_parser("dndc", help="DNDC seam — validate / inspect")
    dsub = p_dndc.add_subparsers(dest="dndc_cmd", required=True)

    p_list = dsub.add_parser("list-presets", help="list available crop presets")
    p_list.set_defaults(_cmd=_cmd_dndc_list)

    p_val = dsub.add_parser("validate", help="validate a DndcSeamInputs JSON file")
    p_val.add_argument("json_path")
    p_val.set_defaults(_cmd=_cmd_dndc_validate)

    p_tf = dsub.add_parser(
        "to-forcing",
        help="materialise a Forcing for inspection (writes summary)",
    )
    p_tf.add_argument("json_path")
    p_tf.add_argument(
        "--sim-times",
        type=str,
        default="0,1,2,3,4",
        help="comma-separated days; default 0,1,2,3,4",
    )
    p_tf.set_defaults(_cmd=_cmd_dndc_to_forcing)

    # ----- soil (M2) ---------------------------------------------------
    p_soil = rsub.add_parser("soil", help="soil-library / PTF")
    ssub = p_soil.add_subparsers(dest="soil_cmd", required=True)

    p_ptf = ssub.add_parser("ptf", help="texture → van Genuchten parameters")
    p_ptf.add_argument("--texture", required=True,
                       help="comma-separated sand=N,silt=M,clay=K (percent)")
    p_ptf.add_argument("--bd", type=float, default=None,
                       help="bulk density g/cm³ (optional; enables ROSETTA H2)")
    p_ptf.add_argument("--theta-33", type=float, default=None)
    p_ptf.add_argument("--theta-1500", type=float, default=None)
    p_ptf.add_argument("--om", type=float, default=None,
                       help="organic matter pct")
    p_ptf.add_argument("--method", default="rosetta3_auto",
                       choices=["rosetta3_auto", "carsel_parrish", "wosten",
                                "rosetta3_h1", "rosetta3_h2", "rosetta3_h3",
                                "rosetta3_h4"])
    p_ptf.set_defaults(_cmd=_cmd_soil_ptf)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hydrus",
        description="HYDRUS-port unified entry: 1D / 2D / 3D Richards.",
    )
    sub = p.add_subparsers(dest="kind", required=True,
                           metavar="{1d,2d,3d,run,test,scenario,convert,research}")

    # ----- 1d ---------------------------------------------------------
    p1d = sub.add_parser(
        "1d",
        help="HYDRUS-1D vertical column (Selector.in + Profile.dat)",
    )
    p1d.add_argument("input_dir", type=Path,
                     help="Directory containing Selector.in and Profile.dat")
    p1d.add_argument("-o", "--output-dir", type=Path, default=None)
    p1d.add_argument("--csv", action="store_true",
                     help="Emit tidy long-format CSV next to each .OUT")
    p1d.set_defaults(func=_run_1d)

    # ----- 2d ---------------------------------------------------------
    p2d = sub.add_parser(
        "2d",
        help="SWMS_2D 2D Richards / solute transport",
    )
    p2d.add_argument("input_dir", type=Path,
                     help="Directory containing SELECTOR.IN and GRID.IN")
    p2d.add_argument("-o", "--output-dir", type=Path, default=None)
    p2d.add_argument("--quiet", action="store_true",
                     help="Suppress per-step progress output")
    p2d.add_argument("--vtk", action="store_true",
                     help="Also write VTK time series next to text outputs")
    p2d.add_argument("--anderson", action="store_true",
                     help="Anderson(m) Picard acceleration (experimental)")
    p2d.add_argument("--anderson-m", type=int, default=3,
                     help="Anderson buffer depth (default 3)")
    p2d.add_argument("--banded", action="store_true",
                     help="Fortran banded-Gauss solver (else SciPy spsolve)")
    p2d.add_argument("--csv", action="store_true",
                     help="Emit tidy long-format CSV next to each .out")
    p2d.set_defaults(func=_run_2d)

    # ----- 3d ---------------------------------------------------------
    p3d = sub.add_parser(
        "3d",
        help="3D Richards (scikit-fem). With no args, runs box demo.",
    )
    p3d.add_argument("input_dir", type=Path, nargs="?", default=None,
                     help="Reserved for future mesh-driven scenarios")
    p3d.add_argument("-o", "--output-dir", type=Path, default=None)
    p3d.add_argument("--csv", action="store_true",
                     help="Emit tidy long-format CSV from the .pvd VTU series")
    p3d.set_defaults(func=_run_3d)

    # ----- test -------------------------------------------------------
    ptest = sub.add_parser(
        "test",
        help="Run smoke tests for one or all simulation paths",
    )
    ptest.add_argument(
        "target", nargs="?", default="all",
        choices=["1d", "2d", "3d", "roundtrip", "cases", "all"],
        help="Which test to run (default: all)",
    )
    ptest.set_defaults(func=_run_test)

    # ----- scenario ---------------------------------------------------
    pscen = sub.add_parser(
        "scenario",
        help="Read or write a SELECTOR.IN as JSON (for the GUI editor)",
    )
    pscen.add_argument("action", choices=["read", "write"])
    pscen.add_argument("input_dir", type=Path,
                       help="Directory containing SELECTOR.IN")
    pscen.add_argument("--json", type=Path, default=None,
                       help="Path to JSON to write (else stdin)")
    pscen.add_argument("--out", type=Path, default=None,
                       help="Override output SELECTOR.IN path")
    pscen.set_defaults(func=_run_scenario)

    # ----- run (canonical case file) -----------------------------------
    prun = sub.add_parser(
        "run",
        help="Run a canonical .json scenario (any dimension)",
    )
    prun.add_argument("case", type=Path, help="Canonical scenario JSON file")
    prun.add_argument("-o", "--output-dir", type=Path, default=None)
    prun.add_argument("--quiet", action="store_true")
    prun.add_argument("--csv", action="store_true",
                      help="Emit tidy long-format CSV next to outputs")
    prun.set_defaults(func=_run_case)

    # ----- convert ----------------------------------------------------
    pconv = sub.add_parser(
        "convert",
        help="Convert legacy .OUT files in a directory to tidy long CSV",
    )
    pconv.add_argument("out_dir", type=Path,
                       help="Directory containing .OUT files to convert")
    pconv.add_argument("--input-dir", type=Path, default=None,
                       help="Path to inputs (for GRID.IN, optional)")
    pconv.set_defaults(func=_run_convert)

    # ----- research (M1 dndc + M2 soil/ptf) --------------------------
    _build_research_subparser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Research subcommands (M1 dndc + M2 soil) set `_cmd`; legacy uses `func`.
    if hasattr(args, "_cmd"):
        return args._cmd(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
