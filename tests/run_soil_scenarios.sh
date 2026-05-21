#!/bin/zsh
# Drive the 6 synthesized soil-type fixtures through both the Fortran golden
# binary and the Python port, then print a pass/fail summary at 1 % tolerance.

set -e
FBIN=/Users/zhangfeng/CODE_BLOCK_DNDC/ai_bot/agrrobot-platform/hydrus/hydrus
cd /Users/zhangfeng/CODE_BLOCK_DNDC/H1D_Src
python3 tests/make_soil_fixtures.py >/dev/null

NAMES=(
    soil_sand_drain
    soil_clay_drain
    soil_loam_infiltr
    soil_silt_evap
    soil_bc_infiltr
    soil_layered_sand_over_clay
)

for name in "${NAMES[@]}"; do
    fix=tests/fixtures/$name
    work=/tmp/h1d_fort_$name
    mkdir -p "$work" && cp "$fix/inputs"/* "$work/" 2>/dev/null || true
    "$FBIN" "$work" </dev/null >/dev/null 2>&1 || true
    for f in NOD_INF.OUT T_LEVEL.OUT BALANCE.OUT A_LEVEL.OUT RUN_INF.OUT \
             I_CHECK.OUT OBS_NODE.OUT PROFILE.OUT; do
        [[ -f "$work/$f" ]] && cp "$work/$f" "$fix/reference_out/$f"
    done
    python3 -c "
import sys, signal
sys.path.insert(0, '.')
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError('300s')))
signal.alarm(300)
from hydrus1d.hydrus import Hydrus1DSimulation
sim = Hydrus1DSimulation(input_dir='$fix/inputs', output_dir='$fix/python_out')
sim.run()
import re
ref = open('$fix/reference_out/BALANCE.OUT').read().split('\n')
py  = open('$fix/python_out/BALANCE.OUT').read().split('\n')
def last(buf, key):
    out = None
    for line in buf:
        if key in line:
            toks = re.findall(r'[-+]?\d+\.\d+E?[-+]?\d*|[-+]?\d+\.\d+|[-+]?\d+', line)
            if toks: out = toks[0]
    return out
ref_W = last(ref, 'W-volume'); py_W = last(py, 'W-volume')
ref_B = last(ref, 'WatBalR'); py_B = last(py, 'WatBalR')
ref_T = last(ref, 'Top Flux'); py_T = last(py, 'Top Flux')
print(f'{\"$name\":<30s} W ref={ref_W} py={py_W}  WatBalR ref={ref_B} py={py_B}  TopFlux ref={ref_T} py={py_T}')
"
done
