#!/bin/zsh
# Batch-test the Python HYDRUS-1D port against the Fortran golden binary on
# all the agrrobot fixtures we know about. For each fixture it
#  (1) copies inputs into tests/fixtures/<name>/inputs/
#  (2) runs the Fortran binary → tests/fixtures/<name>/reference_out/
#  (3) runs Python → tests/fixtures/<name>/python_out/
#  (4) reports BALANCE.OUT / T_LEVEL.OUT / NOD_INF.OUT diffs.

set -e

FORTRAN_BIN="/Users/zhangfeng/CODE_BLOCK_DNDC/ai_bot/agrrobot-platform/hydrus/hydrus"
FIXTURE_BASE="/Users/zhangfeng/CODE_BLOCK_DNDC/H1D_Src/tests/fixtures"
AGRROBOT_RUNS="/Users/zhangfeng/CODE_BLOCK_DNDC/ai_bot/agrrobot-platform/agrrobot-data/hydrus/runs"
cd /Users/zhangfeng/CODE_BLOCK_DNDC/H1D_Src

# Fixture name → source uuid in agrrobot
typeset -A FIXTURES=(
    scenario_3d_water  39b092b7-2787-4efe-a94b-4f7129094a46
    scenario_3d_chem   ab1ca755-5197-4eb0-a366-3534ffdc3299
    field_backed_a     00050835-f3bc-4072-a292-e75e4e555add
    field_backed_b     46a5c974-c066-4214-920a-ad1e2374b458
    field_backed_c     b2ef0b9c-e4b0-4c5b-ae21-3a098d3a60dc
    what_if_evap_b     7165dafa-af3c-43c6-b636-117b7aba23b1
)

stage_fixture() {
    local name=$1 uuid=$2
    local src="$AGRROBOT_RUNS/$uuid"
    local dst="$FIXTURE_BASE/$name"
    if [[ -d "$dst" ]]; then
        echo "  [stage] $name already exists, skipping"
        return
    fi
    mkdir -p "$dst/inputs" "$dst/reference_out" "$dst/python_out"
    for f in Selector.in Profile.dat Atmosph.in ATMOSPH.IN Meteo.in Hydrus1d.dat; do
        [[ -f "$src/$f" ]] && cp "$src/$f" "$dst/inputs/$f"
    done
    # Normalize Atmosph filename
    if [[ -f "$dst/inputs/ATMOSPH.IN" ]] && [[ ! -f "$dst/inputs/Atmosph.in" ]]; then
        cp "$dst/inputs/ATMOSPH.IN" "$dst/inputs/Atmosph.in"
    fi
}

run_fortran() {
    local name=$1
    local dst="$FIXTURE_BASE/$name"
    local workdir="/tmp/h1d_fort_$name"
    rm -rf "$workdir" 2>/dev/null || true
    mkdir -p "$workdir"
    cp "$dst/inputs/"* "$workdir/"
    cd "$workdir"
    "$FORTRAN_BIN" "$workdir" </dev/null >/dev/null 2>&1 || true
    cd /Users/zhangfeng/CODE_BLOCK_DNDC/H1D_Src
    for f in NOD_INF.OUT T_LEVEL.OUT BALANCE.OUT A_LEVEL.OUT RUN_INF.OUT \
             I_CHECK.OUT OBS_NODE.OUT PROFILE.OUT; do
        [[ -f "$workdir/$f" ]] && cp "$workdir/$f" "$dst/reference_out/$f"
    done
}

run_python() {
    local name=$1
    local dst="$FIXTURE_BASE/$name"
    python3 - <<PYEOF
import sys, signal, time
sys.path.insert(0, '.')
signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError('120s')))
signal.alarm(120)
from hydrus1d.hydrus import Hydrus1DSimulation
t0 = time.time()
sim = Hydrus1DSimulation(input_dir='$dst/inputs', output_dir='$dst/python_out')
sim.run()
print(f"  [python] $name done in {time.time()-t0:.1f}s  t={sim.state.t}")
PYEOF
}

echo "============================================================"
echo "HYDRUS-1D Python port: scenario batch test"
echo "============================================================"
for name uuid in "${(@kv)FIXTURES}"; do
    echo ""
    echo "▶ $name  ($uuid)"
    stage_fixture "$name" "$uuid"
    run_fortran "$name"
    run_python "$name"
    echo "  ── compare (rtol=1e-2) ──"
    python3 tests/compare_outputs.py "$FIXTURE_BASE/$name" --rtol 1e-2 2>&1 | tail -3
done
echo ""
echo "============================================================"
echo "Done.  Detailed reports in tests/fixtures/<name>/python_out/"
echo "============================================================"
