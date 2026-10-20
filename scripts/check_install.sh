#!/usr/bin/env bash
# Check a new machine before the first live scan: GPU and Docker, the image, the GPU copy check,
# and a replay of a recording through the full pipeline. No detector needed (except for the
# optional DCU reachability check). Run from the repository root:
#
#   scripts/check_install.sh                                   # the basic checks
#   RECORDING=/data/scan.dstream SCAN="1024 1024" BINNED=1 scripts/check_install.sh
#   DCU=tcp://10.0.0.10:31001 scripts/check_install.sh         # also: can we reach the DCU?
#
# Environment: IMAGE (default realtime-stem:latest), GPU (default 0: which GPU the pipeline uses,
# as numbered by nvidia-smi), RECORDING + SCAN (+ BINNED=1 for 96x96 frames) for a full-size
# replay, DCU for the reachability and link-speed check. Every step prints PASS / FAIL / SKIP.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
IMAGE="${IMAGE:-realtime-stem:latest}"
GPU="${GPU:-0}"
NAME=realtime_stem_check
OUT="$REPO/out/check_install"
FAILED=0

pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILED=$((FAILED + 1)); }
skip() { echo "SKIP  $*"; }
in_image() {   # run python3 in the image on the chosen GPU, the repository mounted at /work
  docker run --rm --gpus "device=$GPU" --ulimit stack=33554432:33554432 --network host \
    -v "$REPO":/work -w /work --entrypoint python3 "$IMAGE" "$@"
}
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT

echo "== 1. GPU and driver"
if nvidia-smi --query-gpu=index,name,driver_version,memory.total,mig.mode.current \
     --format=csv,noheader; then
  pass "nvidia-smi"
  echo "      (GPU $GPU will be used; other users' processes on it:)"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | sed 's/^/      /'
else
  fail "nvidia-smi (is the NVIDIA driver installed?)"
fi

echo "== 2. Docker can use the GPU, and the image exists"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  pass "image $IMAGE"
else
  fail "image $IMAGE not found: build it (docker build -t $IMAGE -f docker/Dockerfile .) or docker load it"
  echo "Stopping: the remaining checks need the image."; exit 1
fi
if in_image -c "$(cat <<'PY'
import torch, dectris_holoscan_ops, holoscan
from realtime_stem import __version__
p = torch.cuda.get_device_properties(0)
ops = getattr(dectris_holoscan_ops, "__version__", "before 2.0")
print(f"      {p.name}, compute capability {p.major}.{p.minor}, {p.total_memory / 2**30:.0f} GiB")
print(f"      torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"holoscan {getattr(holoscan, '__version__', '?')}")
print(f"      realtime-stem {__version__}, operators {ops}")
torch.ones(1, device="cuda").add_(1)
PY
)"; then
  pass "the image runs on GPU $GPU"
else
  fail "the image does not run on GPU $GPU (NVIDIA container toolkit installed? docker run --gpus works?)"
fi

echo "== 3. Copies between pinned memory and the GPU"
if in_image -m tools.check_gpu_copies; then pass "GPU copies"; else fail "GPU copies"; fi

replay() {   # replay DSTREAM through the pipeline: replay LABEL DSTREAM SCAN EXTRA_ARGS...
  local label=$1 dstream=$2 scan=$3; shift 3
  local log="$OUT/$label.log"
  mkdir -p "$OUT/$label"
  cleanup
  docker run -d --name "$NAME" --gpus "device=$GPU" --network host \
    --ulimit stack=33554432:33554432 -v "$REPO":/work -v "$OUT/$label":/out -w /work \
    --entrypoint python3 "$IMAGE" -m realtime_stem.apps.run --algorithm parallax \
    --endpoint tcp://127.0.0.1:5599 --scan $scan --n-connections 3 --out-dir /out "$@" \
    >/dev/null || { fail "$label: pipeline did not start"; return; }
  sleep 8
  docker run --rm --network host -v "$REPO":/work -v "$(dirname "$dstream")":/rec:ro -w /work \
    --entrypoint python3 "$IMAGE" -m tools.replay_to_zmq "/rec/$(basename "$dstream")" \
    --endpoint tcp://127.0.0.1:5599 --fast >/dev/null 2>&1 &
  local feeder=$!
  if ! timeout 300 docker wait "$NAME" >/dev/null; then
    docker kill -s INT "$NAME" >/dev/null 2>&1; sleep 5
  fi
  docker logs "$NAME" > "$log" 2>&1
  wait $feeder 2>/dev/null
  grep -h "TOTAL frames\|COMPLETE" "$log" | sed 's/^/      /'
  if grep -q "\[sink\] COMPLETE" "$log"; then pass "$label"; else fail "$label (log: $log)"; fi
}

echo "== 4. Replay of the bundled sample through the full pipeline"
replay sample "$REPO/examples/data/sample.dstream" "32 32" --binned --batch-size 128

echo "== 5. Full-size replay of a real recording"
if [ -n "${RECORDING:-}" ] && [ -n "${SCAN:-}" ]; then
  ARGS=(); [ "${BINNED:-0}" = 1 ] && ARGS+=(--binned)
  replay recording "$RECORDING" "$SCAN" "${ARGS[@]}"
  echo "      (the rate above should be well over the detector's frame rate, e.g. > 120 kHz)"
else
  skip "set RECORDING=<file.dstream> and SCAN=\"Ny Nx\" (and BINNED=1 for 96x96 frames)"
fi

echo "== 6. Can this machine reach the DCU?"
if [ -n "${DCU:-}" ]; then
  hostport=${DCU#tcp://}
  host=${hostport%:*}
  if timeout 3 bash -c "exec 3<>/dev/tcp/$host/${hostport##*:}" 2>/dev/null; then
    pass "DCU at $hostport accepts connections"
  else
    fail "DCU at $hostport does not accept connections (address, port, stream enabled, firewall?)"
  fi
  # The link towards the DCU: 120 kHz of binned beam-on frames is about 3 Gbit/s of data.
  dev=$(ip route get "$host" 2>/dev/null | grep -o "dev [^ ]*" | head -1 | cut -d" " -f2)
  speed=$(cat "/sys/class/net/$dev/speed" 2>/dev/null)
  if [ -z "$dev" ] || [ "$dev" = lo ]; then
    skip "link speed (no route to $host, or a local address)"
  elif ! [[ "$speed" =~ ^[0-9]+$ ]] || [ "$speed" = 0 ]; then
    skip "link speed of $dev unknown (virtual or bonded interface?)"
  elif [ "$speed" -ge 10000 ]; then
    pass "link to the DCU: $dev at $((speed / 1000)) Gbit/s"
  else
    fail "link to the DCU: $dev at $speed Mbit/s; 120 kHz needs 10 Gbit/s or more"
  fi
  echo "      (make sure nothing else on the network reads this stream during the scans)"
else
  skip "set DCU=tcp://<DCU_IP>:31001"
fi

echo
if [ "$FAILED" = 0 ]; then echo "All checks passed."; else echo "$FAILED check(s) FAILED."; fi
exit $(( FAILED > 0 ))
