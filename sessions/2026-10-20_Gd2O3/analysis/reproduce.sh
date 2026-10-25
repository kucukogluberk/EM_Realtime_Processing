#!/bin/bash
# Everything of the 2026-10-20 Gd2O3 session from its two recordings, in order. ~50 min on an M4
# Pro, most of it quantem (step 2; ~10 GB RAM for the unbinned scan). Set PY / QUANTEM_PY to other
# environments, DEVICE=cuda (or cpu) off a Mac. Each step can also be run on its own.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SESSION=$(cd "$HERE/.." && pwd)
REPO=$(cd "$SESSION/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
QPY=${QUANTEM_PY:-$REPO/../quantem-upstream/.venv/bin/python}
cd "$REPO"

echo "== 1. the standard analysis: calibration (C10 refined on the lattice), the reconstructions"
echo "      (×1/2/4/8, the support focus, the live flags, the frame-mean and DC variants),"
echo "      figures/overview.jpg, figures/upsampling_*.jpg, summary.json"
$PY scripts/session_report.py "$SESSION" --device "${DEVICE:-auto}"

echo "== 2. quantem: DirectPtychography and the montage, and its residual loss → recon/quantem/"
rm -rf "$SESSION/recon/quantem"
(cd "$HERE" && $PY export_for_quantem.py && $QPY quantem_recons.py binned \
    && $QPY quantem_recons.py unbinned && $QPY quantem_loss.py)

echo "== 3. continuous modes on a looped copy with drift (the loop goes to out/, 0.7 GB)"
REC=$($PY -c "import json; print(json.load(open('$SESSION/session.json'))['recordings']['binned']['file'])")
[ -f out/binned_drift_x4.dstream ] || $PY -m tools.repeat_dstream "$SESSION/$REC" \
    out/binned_drift_x4.dstream --passes 4 --scan 256 256 --drift 3 4
mkdir -p "$SESSION/figures/full"
$PY scripts/compare_accumulate.py out/binned_drift_x4.dstream $(cat "$SESSION/calibration/binned.flags") \
    --upsampling 2 --device "${DEVICE:-mps}" --out "$SESSION/figures/full/continuous_modes.png"

echo "== 4. the session's own figures → figures/full/, and the downscaled ones kept in git → figures/"
(cd "$HERE" && $PY figures.py > /dev/null && $PY frame_mean.py && $PY midscan.py \
    && $PY publish_figures.py)
