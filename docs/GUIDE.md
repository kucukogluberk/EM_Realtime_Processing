# realtime-stem — technical guide

The full description: architecture, build, usage, known issues. For the short version (what is
where, install, run) see the [README](../README.md).

Real-time STEM data processing at full detector frame rate. Detector frames stream from a DECTRIS
detector control unit (DCU), are GPU-decompressed on an NVIDIA Holoscan ingest, folded into a
reconstruction as they arrive, and streamed out live over ZeroMQ — so the reconstruction fills in
*during* the scan rather than only at the end.

The worked example is **parallax / tilt-corrected bright field (tcBF)**, running at 120 kHz on
binned ARINA data. The pipeline is deliberately modular so a new algorithm is a drop-in.

## Architecture

Three stages with narrow interfaces, so each can be understood and changed on its own:

```
   DCU ─(ZeroMQ)─►  ingest  ─(batches)─►  processing  ─(reconstruction)─►  stream-out  ─(ZeroMQ)─►  subscribers
                 ZmqRxOp →              Accumulator                     LivePublisher            viewer / probe / …
                 DecompressGpuOp        .process(frames, ids)
                 (C++/CUDA)             .result()
```

- **ingest** (`realtime_stem/ingest/`) — receives raw frames and bslz4-decompresses them on the
  GPU using two compiled Holoscan C++ operators, then hands each decoded batch to an `Accumulator`
  as a zero-copy CUDA tensor. Algorithm-agnostic.
- **processing** (`realtime_stem/processing/`) — turns batches into a reconstruction behind two
  small contracts (`processing/base.py`): `Accumulator` (the hot path) and `Algorithm` (the
  wiring). Parallax lives in `processing/parallax/`.
- **stream-out** (`realtime_stem/streaming/`) — publishes the reconstruction-so-far over ZeroMQ
  (PUB, lossy, non-blocking). A matplotlib viewer is just one example subscriber; the product is
  the stream.

`apps/run.py` wires the three together and owns only transport flags; it never mentions parallax.

## Prerequisites

The pipeline runs inside a Docker image that carries Holoscan, a CUDA build of PyTorch, nvcomp
and the two compiled C++/CUDA Holoscan operators `ZmqRxOp` and `DecompressGpuOp`. **This
repository contains everything needed to build it**: the operator sources are in `operators/`
(see `operators/README.md`) and `docker/Dockerfile` builds the complete image on top of NVIDIA's
public base image (`nvcr.io/nvidia/clara-holoscan/holoscan:v4.4.0-cuda12-dgpu`, which supplies
Holoscan 4.4 and PyTorch 2.11). From the repository root:

```bash
docker build -t realtime-stem:latest -f docker/Dockerfile .
```

The build compiles the CUDA kernels, so it takes a few minutes.

### What the operators do

They originate in the public, MIT-licensed
[electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d) repository. The version
in `operators/` extends it with what this pipeline needs (the upstream build alone is hard-wired to
96×96 uint16 frames and a single TCP connection):

1. runtime `detector_width` / `detector_height` / `dtype` parameters, validated against the stream
   (a mismatch stops with an error instead of emitting wrong-sized garbage);
2. the bslz4 block size read from each frame's header (required for uint32 streams);
3. an `n_connections` parameter (the DCU paces each TCP connection, so 120 kHz needs 3);
4. a clean end of scan (since 2.0.0): after `end` the receiver drains all connections, flushes
   the last partial batch and stops only itself, so everything received is processed; the
   connection to the DCU stays open across the scans of a `--continuous` session;
5. series handling and checks (since 2.0.0): the `series_id` keeps one run to one series, the
   `start` message is compared with the configuration (`expected_images` = Ny×Nx), every image's
   compressed data is checked on the GPU, and the receiver waits with a 100 ms timeout so the
   application can always be stopped.

`realtime_stem/ingest/holoscan_graph.py` passes those parameters and refuses to start on an image
built from the upstream operators.

### Host requirements

`run.py` prints the operators' version at start-up (`operators 2.0.1`). "before 2.0" means the
image was built from older sources and lacks the fixes listed in `CHANGELOG.md`.

- Linux, Docker, the NVIDIA container toolkit (`--gpus all`), and an NVIDIA driver that supports
  CUDA 12.x (the `cuda12-dgpu` base image; a `cuda13` base exists for newer driver stacks). The
  `dgpu` base covers x86_64 and aarch64 hosts with a discrete GPU (IGX Orin dGPU, GH200). Jetson
  integrated GPUs would need the `igpu` base and are untested here.
- **GPU architecture.** The CUDA kernels are compiled for the architectures named at image build
  time. The default in `docker/Dockerfile` is `80;86;89;90`: A100 (sm_80), RTX 30xx / A6000
  (sm_86), RTX 40xx / RTX 6000 Ada / L4 / L40 (sm_89) and H100 / H200 (sm_90). Newer GPUs (Blackwell) run
  via PTX just-in-time compilation from compute_90 (slower first launch, otherwise fine). **Older
  GPUs (V100 sm_70, T4 sm_75) fail with "no kernel image is available"** unless their
  architecture is added with `--build-arg CUDA_ARCHITECTURES=...`.
- Validated on an NVIDIA IGX Orin (aarch64) with an RTX 6000 Ada. On x86_64 the image builds
  (kernels for sm_80 to sm_90), its operators import and the unit tests pass; a run on an x86_64
  GPU is checked with `scripts/check_install.sh`. NVIDIA's CUDA compiler does not run under
  emulation, so build the x86_64 image on an x86_64 machine.

The processing engine and streaming transport are pure PyTorch / numpy / pyzmq; PyTorch and the
Holoscan ops come from the container, so nothing GPU-specific is pip-installed here.

## Quick start — offline replay demo (no detector)

Replay a bundled synthetic capture through the whole pipeline and watch the reconstruction wipe in:

```bash
scripts/run_replay_demo.sh
```

It starts `run.py` in the container, a viewer, and a replay feeder that streams
`examples/data/sample.dstream` (a 32×32 scan of 96×96 uint16 frames, 1024 frames, real bslz4
compression in the real DECTRIS message framing) over localhost. The viewer and the feeder run on
the host's `python3`, which needs numpy, pyzmq, matplotlib and cbor2:

```bash
pip install -e ".[viz,replay]"
```

Make your own test capture with `python -m tools.make_dstream 96 96 uint16 1024 --out
examples/data/sample.dstream` (needs the decode deps, `pip install -e ".[decode]"`, i.e. cbor2,
bitshuffle and dectris-compression, in a host environment; the image does not carry them).

## Live usage (against a detector)

Run the pipeline inside the container, pointing it at the DCU's ZMQ stream:

```bash
docker run --rm --network host --gpus all --ulimit stack=33554432:33554432 \
  -v "$PWD":/work -v "$PWD/out":/out -w /work --entrypoint python3 \
  realtime-stem:latest -m realtime_stem.apps.run \
    --algorithm parallax \
    --endpoint tcp://<DCU_IP>:31001 \
    --binned --scan 256 256 --batch-size 2048 --n-connections 3 \
    --defocus <Å> --rotation <degrees> --upsampling 2 \
    --live-endpoint tcp://127.0.0.1:5601 --continuous
```

On the host, watch it fill in:

```bash
python examples/viewer.py tcp://127.0.0.1:5601
```

`--continuous` re-arms after each scan's end (one container for a whole session). Results
(`parallax_recon.npy`, `parallax_shifts.npy`) land in `out/`; plot a saved one with
`python examples/plot_result.py out/parallax_recon.npy`. See `run.py --help` for all flags; the
parallax tuning knobs (`--defocus`, `--rotation`, `--energy`, `--semiangle`, `--angular-sampling`,
`--scan-sampling`, `--center`, …) use the same units live as offline.

Three parameters decide whether the image is sharp, and none of them comes from the stream:
`--defocus` (C10 in Å, the microscope's readout is the starting point), `--rotation` (the
scan-to-detector rotation in degrees) and `--center` (the bright-field disk's centre on the
detector in pixels, e.g. from py4DSTEM's `get_probe_size`). Tune them offline against a replayed
capture; the flags and units are identical live.

### Recording a scan

A recording lets you tune `--defocus`, `--rotation` and `--center` offline, as often as needed.
Stop the pipeline first (only one program may read the stream), then:

```bash
python3 tools/record_stream.py capture --ip <DCU_IP> --out scan01.dstream
```

and start the scan. The recorder (host Python with `pyzmq` and `cbor2`) stops by itself at the
end of the scan and reports whether every frame arrived (`COMPLETE`). Replay it into the pipeline
with `tools/replay_to_zmq.py`; keep the microscope settings of each recording next to it.

## Known issues and limitations

- **The run reports whether the scan is complete.** After every scan: `[sink] COMPLETE: all N
  scan positions received`, or `WARNING: INCOMPLETE SCAN: … (k missing, d frames for positions
  that already had one)`. The result is saved either way. Causes of an incomplete scan: a second
  reader of the DCU stream, a scan started too soon after the previous one, frames left over from
  a crashed run (the log then says the series' start message was not seen), or detector-side
  drops.
- **Scans in quick succession.** Between scans the pipeline is rebuilt (about a second). The
  connection to the DCU stays open meanwhile, so the next scan's frames wait in its receive queue
  (up to 50,000 messages per connection) and in the DCU's buffer; none are lost when scans follow
  each other directly (tested at 120 kHz with no gap). Frames of the next series are never mixed
  into the current one: the receiver tells them apart by `series_id`.
- **The batch size is capped at compile time** (`kMaxBatchSize` in
  `operators/cpp/common/constants.h`, 2048). Larger `--batch-size` values are clamped, with a
  warning in the log. Any smaller size works, but small batches are slower (see the next point).
- **Batch size and rate.** Replaying a 1024×1024 scan of binned 96×96 beam-on frames on an RTX 6000
  Ada (one connection): batch 512 → 69 kHz, 1024 → 114 kHz, 2048 → 160 kHz, all frames received.
  The default is therefore 2048 (the cap); GPU memory is reserved for the cap whatever the batch.
  These numbers are for the 4-thread `MultiThreadScheduler`: the limit at small batches is its
  polling (idle workers sleep up to 5 ms), not the GPU.
- **Scheduler.** Since 2.0.2 the default is Holoscan's single-threaded scheduler
  (`--scheduler-threads 0`): it runs the three operators one after another without sleeping, while
  ZeroMQ keeps receiving in its own I/O thread. On the test machine (IGX Orin, RTX 6000 Ada) it was faster in every test (1024²
  binned: full speed 166–168 kHz against 149–151 with 4 threads; paced at 120 kHz 120.5 kHz
  against 113.5, i.e. 4 threads end a scan about 0.5 s late; busy machine 117–120 against 96–114)
  and uses less CPU. `--scheduler-threads 4` gives the `MultiThreadScheduler` of 1.0.0–2.0.1.
- **Connections and network.** The DCU paces each TCP connection to about 60k beam-on binned
  frames per second, so 120 kHz needs 3 connections (the default since 2.0.1; two connections
  cap near 112 kHz, and the DCU then drops frames once its buffer of about 65k frames is full).
  The link from the DCU's data interface must be 10 Gbit/s or faster: 120 kHz of binned beam-on
  frames is about 370 MB/s (3 Gbit/s), so a 1 Gbit/s link caps near 40 kHz. Unbinned frames have
  four times as many pixels but run at most at 30 kHz (the DCU's shortest unbinned frame time is
  33.3 µs), so their data rate is of the same order. `DCU=tcp://<DCU_IP>:31001
  scripts/check_install.sh` prints the link speed.
- **Busy machine.** With 8 of the test machine's 12 cores kept busy by other programs, every scan was
  still complete. With the 4-thread scheduler the pipeline fell up to 2 s behind by the end of a
  1024² scan at a paced 120 kHz (the backlog waits in the receive queues, room for 150k frames,
  then in the DCU's buffer, which is finite); the single-threaded default kept up (117–120 kHz).
  On a shared server, also keep the container on cores of its own (`docker run --cpuset-cpus`),
  on the CPU socket next to the GPU.
- **uint32 streams are handled in code but have not been run live** through this sink. The DCU
  sends uint32 when the frame time is long (a few kHz); `--dtype uint32` configures the ops and the
  engine, and the engine's uint32 path is unit-tested, but the only live validation is uint16.
- **DECTRIS Stream2 only.** The ingest understands DECTRIS CBOR messages with bslz4 payloads (the
  DCU's stream compression must be `bslz4`). A different detector needs a different ingest; the
  processing stage plugs in behind the `Accumulator` contract and needs no change.
- **Start the pipeline before the scan.** While no receiver is connected, the DCU drops the
  frames it streams (a pipeline started during a scan receives only the rest of it, and the run
  reports the scan as incomplete). The DCU counts
  them: `GET http://<DCU_IP>/stream/api/1.8.0/status/dropped`. In `--continuous` mode the
  connection stays open between scans.
- **One consumer per DCU stream.** The DCU's PUSH socket load-balances across every connected PULL
  peer, so a second listener (or a crashed one's leftover connection) silently takes a share of
  the frames. The run's completeness report shows it as missing positions.
- **Frame order.** Each frame is placed by its `image_id` (raster index), so the order in which
  frames arrive does not matter, but the scan must deliver exactly one frame per position in raster
  order of `image_id`. Extra frames (e.g. flyback) shift the mapping; the `start` check warns when
  the detector announces more images than `--scan` has positions.

## Adding an algorithm

The pipeline depends only on the two contracts in `realtime_stem/processing/base.py`, so a new
method (e.g. strain mapping) is self-contained:

1. Implement an `Accumulator`: `process(frames, ids)` folds a batch into the reconstruction-so-far;
   `result()` returns it; expose `detector_shape` / `scan_shape`.
2. Implement an `Algorithm`: declare its CLI flags, build the accumulator from them, save results.
3. Register it in `realtime_stem/processing/registry.py`.

`run.py --algorithm <name>` and the ingest graph need no changes. `processing/parallax/` is the
reference implementation to copy from.

## Layout

```
realtime_stem/
  ingest/holoscan_graph.py     ZmqRxOp → DecompressGpuOp → GenericSinkOp (drives any Accumulator)
  processing/base.py           the Accumulator + Algorithm contracts
  processing/registry.py       --algorithm name → Algorithm
  processing/parallax/         engine (geometry + accumulator), config, algorithm
  streaming/live_stream.py     pack/unpack + PUB/SUB (CONFLATE) transport
  apps/run.py                  the runner (transport flags + continuous/rounds loop)
tools/                         .dstream format, stream recorder, replay feeder, synthetic-capture generator
examples/                      viewer, headless probe, static result plot
tests/                         parallax oracle tests, streaming transport tests, runner CLI tests
operators/                     the C++/CUDA Holoscan operators (ZmqRxOp, DecompressGpuOp), MIT
docker/Dockerfile              builds the complete image: Holoscan base + operators + pyzmq
docs/GUIDE.md                  this guide
```

## Tests

Pure-Python (no detector, no container). From the repo root, in an environment with numpy + torch
+ pyzmq:

```bash
python -m pytest tests/
```

The GPU-parity test runs where CUDA is available and skips cleanly otherwise. The parallax tests
pin the engine against a float64 transcription of the reference math and pin the CLI unit
conversions.

## Credits and relation to the reference

The real-time parallax / tcBF algorithm is the work of Georgios Varnavides. This implementation
follows his streaming-parallax reference and reproduces its output on the public apoferritin
dataset (Zenodo 10.5281/zenodo.18346854) to float32 precision. The scatter wraps `% Ny` and `% Nx`
independently, and the one-pixel bright-field-index correction for quarter-turn rotations is not
applied, because here the disk position is the measured `--center`.

The GPU ingest (`operators/`) is built on the Holoscan operators from
[electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d) (Denis Leshchev,
NVIDIA; MIT, see `operators/LICENSE`) with the generic-geometry, header-driven block size and
multi-connection extensions described under Prerequisites.
