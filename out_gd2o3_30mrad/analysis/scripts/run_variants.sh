#!/bin/bash
# The realtime-stem reconstructions of both recordings with their calibrated flags: upsampling
# 1/2/4/8 × frame mean kept / --subtract-frame-mean / --no-dc-correct → out_gd2o3_30mrad/recon/.
# Python ingest on the Mac (--device mps; cpu or cuda work too). ~2 min. Run from anywhere.
set -euo pipefail
EXP=$(cd "$(dirname "$0")/../.." && pwd)
REPO=$(cd "$EXP/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
DEVICE=${DEVICE:-mps}
cd "$REPO"
for mode in binned unbinned; do
  if [ $mode = binned ]; then src=$EXP/Gd2O3_50.6nmFOV_256x256_30mrad_-30nm_binned_10us.dstream
  else src=$EXP/Gd2O3_50.6nmFOV_256x256_30mrad_-30nm_unbinned.dstream; fi
  flags=$(cat "$EXP/analysis/cal_$mode.flags")
  for u in 1 2 4 8; do
    for variant in "" _sub _nodc; do
      case $variant in _sub) extra=--subtract-frame-mean ;; _nodc) extra=--no-dc-correct ;; *) extra= ;; esac
      out=$EXP/recon/${mode}_U$u$variant
      mkdir -p "$EXP/recon"
      $PY -m realtime_stem.apps.run --ingest python --device "$DEVICE" --source "$src" $flags \
          --upsampling $u $extra --out-dir "$out" > "$out.log" 2>&1
      echo "$mode U$u$variant: $(grep -E 'COMPLETE|INCOMPLETE' "$out.log")"
    done
  done
done
