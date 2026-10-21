#!/bin/bash
# Everything in out_gd2o3_30mrad/analysis from the two recordings, in order. ~45 min on an M4 Pro,
# most of it quantem (step 3; ~10 GB RAM for the unbinned scan). Set PY / QUANTEM_PY to other
# environments, DEVICE=cuda (or cpu) off a Mac. Each step can also be run on its own.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$(cd "$HERE/../.." && pwd)
REPO=$(cd "$EXP/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
QPY=${QUANTEM_PY:-$REPO/../quantem-upstream/.venv/bin/python}
REC=$EXP/Gd2O3_50.6nmFOV_256x256_30mrad_-30nm
ACQ="--scan 256 256 --energy 200000 --semiangle 30 --scan-sampling 1.9766 --rotation-prior 3.75"
cd "$REPO"

echo "== 1. calibration: centre, flip, rotation, C10, C12 → analysis/cal_*.{flags,json,png}"
$PY -m tools.calibrate_parallax ${REC}_binned_10us.dstream $ACQ --angular-sampling 1.066 \
    --out "$EXP/analysis/cal_binned"
$PY -m tools.calibrate_parallax ${REC}_unbinned.dstream $ACQ --angular-sampling 0.533 \
    --out "$EXP/analysis/cal_unbinned"

echo "== 2. realtime-stem reconstructions → recon/"
"$HERE/run_variants.sh"

echo "== 3. quantem: DirectPtychography and the montage, and its residual loss → recon/quantem/"
(cd "$HERE" && $PY export_for_quantem.py && $QPY quantem_recons.py binned \
    && $QPY quantem_recons.py unbinned && $QPY quantem_loss.py)

echo "== 4. continuous modes on a looped copy with drift (the loop goes to out/, 0.7 GB)"
$PY -m tools.repeat_dstream ${REC}_binned_10us.dstream out/binned_drift_x4.dstream --passes 4 \
    --scan 256 256 --drift 3 4
$PY scripts/compare_accumulate.py out/binned_drift_x4.dstream $(cat "$EXP/analysis/cal_binned.flags") \
    --upsampling 2 --device "${DEVICE:-mps}" --out "$EXP/analysis/C_compare_modes.png"

echo "== 5. figures → analysis/*.png, and the downscaled ones kept in git → analysis/figures/"
(cd "$HERE" && $PY figures.py > /dev/null && $PY frame_mean.py && $PY midscan.py \
    && $PY publish_figures.py)
