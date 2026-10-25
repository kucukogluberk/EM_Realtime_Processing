#!/bin/bash
# Everything of the 2026-10-23 Au-grating session from its two recordings (and the live
# snapshots), in order. ~2 min on an M4 Pro. ctf_zeros.py and disk_size.py also read the
# 2026-10-20 session: run its step 1 first. Set PY to another environment, DEVICE=cuda (or cpu)
# off a Mac. Each step can also be run on its own.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SESSION=$(cd "$HERE/.." && pwd)
REPO=$(cd "$SESSION/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
cd "$REPO"

echo "== 1. the standard analysis: calibration (C10 refined on the lattice), the reconstructions"
echo "      (×1/2/4/8, the support focus, the presumed live flags), figures/overview.jpg,"
echo "      figures/upsampling_*.jpg, summary.json"
$PY scripts/session_report.py "$SESSION" --device "${DEVICE:-auto}"

echo "== 2. the session's own checks → figures/full/ and analysis/summary.json, and the"
echo "      downscaled figures kept in git → figures/"
(cd "$HERE" && $PY disk_size.py > /dev/null && $PY scan_step.py > /dev/null \
    && $PY ctf_zeros.py > /dev/null && $PY publish_figures.py)
