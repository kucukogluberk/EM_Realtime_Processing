# realtime-stem

> This repository is frozen at v2.3.0. Development continues in
> [Live4D](https://github.com/electronmicroscopy/live4d), where parallax, its browser viewer, live
> tuning and live OBF run as a branch of the Holoscan SDK 5 pipeline, fed over ZMQ or RDMA (see its
> [parallax guide](https://github.com/electronmicroscopy/live4d/blob/main/docs/parallax/GUIDE.md),
> electronmicroscopy/live4d#15 and electronmicroscopy/live4d#16). In Live4D,
> `python -m live4d.parallax.tools.migrate_flags --flags site.flags` turns `run.py` flags into its
> YAML. This repository keeps the Holoscan 4.4 image, the recorded sessions in `sessions/` and the
> history, and takes no new features.

Real-time STEM processing at full detector frame rate. Frames stream from a DECTRIS ARINA detector
(through its detector control unit, the DCU), are decompressed on an NVIDIA GPU and folded into a
reconstruction **while the scan is running**, so you watch the image fill in during the scan.

The included method is **parallax / tilt-corrected bright field (tcBF)**, with a live
**phase-flipped** image next to it. It runs at 120 kHz on binned (96×96) ARINA frames. Without a
detector or an NVIDIA GPU, the same processing also runs on a laptop, on recorded or simulated
data.

- [`docs/GUIDE.md`](docs/GUIDE.md): the full technical guide
- [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md): from a recording to a live demo (browser viewer,
  calibration, offline reconstruction, looping a scan, a focus-walk demo)
- [`docs/CLI.md`](docs/CLI.md): every command-line option
- [`CHANGELOG.md`](CHANGELOG.md): what changed between versions
- [`AGENTS.md`](AGENTS.md): orientation for AI coding assistants

## What is where

```
realtime_stem/        the Python package
  apps/run.py           the program you run
  ingest/               receives the detector frames
  processing/parallax/  the parallax / tcBF reconstruction
  processing/obf/       live optimum bright field beside it (OBF in the viewer's channel list)
  streaming/            publishes the live image
operators/            the C++/CUDA frame receiver and GPU decompressor
docker/Dockerfile     builds the Docker image
examples/             live viewer, notebook, sample recording
tools/                stream recorder, replay, simulator, detector setup, geometry calibration,
                      looping and refocusing a recording
sessions/             recorded sessions at the microscope, each analysed the same way (data not in git)
scripts/              installation check, demos, the standard session analysis
tests/                unit tests
```

## Requirements

- Linux with an NVIDIA GPU, Docker and the NVIDIA container toolkit (`docker run --gpus`
  must work)
- Python 3.10+ on the host, for the viewer

## Install

Have two things at hand first:

- **The DCU's stream address**: the IP of the DCU's data interface on the network to this
  machine (it can differ from the address of the DCU's web page). Port 31001 by default.
- **Which GPU** to use: its number in `nvidia-smi -L`.

```bash
docker build -t realtime-stem:latest -f docker/Dockerfile .
pip install -e ".[viz,replay]"
GPU=<N> DCU=<DCU_IP> scripts/check_install.sh
```

The check tests the DCU address first and ends with the `--gpus` and `--endpoint` values for the
run command. The image holds the compiled code; the Python package is mounted from this folder at
run time.

## Detector settings

The DCU must stream `cbor` with `bslz4` compression, one image per scan position. Check it (read
only), and set it if needed:

```bash
python -m tools.detector_setup check --ip <DCU_IP> --scan 256 256 --binned
python -m tools.detector_setup apply --ip <DCU_IP> --scan 256 256 --binned --frame-time 8.334e-6
```

## Run

Start the pipeline before the scan:

```bash
docker run --rm --network host --gpus "device=<N>" --ulimit stack=33554432:33554432 \
  -v "$PWD":/work -v "$PWD/out":/out -w /work --entrypoint python3 \
  realtime-stem:latest -m realtime_stem.apps.run \
    --endpoint tcp://<DCU_IP>:31001 --binned --scan 256 256 \
    --energy <eV> --semiangle <mrad> --angular-sampling <mrad/px> --scan-sampling <Å> \
    --defocus <Å> --rotation <degrees> \
    --live-endpoint tcp://127.0.0.1:5601 --continuous --out-dir /out
```

Watch it in a browser: start the viewer in a second terminal and open http://127.0.0.1:8050.

```bash
python -m realtime_stem.streaming.web tcp://127.0.0.1:5601
```

It sends the browser only what is on screen, so large or upsampled images (e.g. 4096²) stay
smooth. The matplotlib window `python examples/viewer.py tcp://127.0.0.1:5601` still works.

Then arm the detector, scan, and disarm, with your usual software or with
`python -m tools.detector_setup arm` / `disarm --ip <DCU_IP>`.

- At start-up it prints the GPU it uses, and `WARNING: cannot reach …` when the DCU address is
  wrong. No frames can arrive until that is fixed.
- The acquisition values (`--energy` in eV, `--semiangle`, `--angular-sampling`,
  `--scan-sampling`, `--scan`, `--defocus`, `--rotation`) have no defaults: a run
  without one stops and lists what is missing. `--center` (the disk centre, px on a raw frame)
  defaults to the frame center. The bright-field disk's radius is not an input; the
  run prints it (semiangle ÷ angular sampling) to compare with a frame. Keep them in a file and pass `$(cat site.flags)`.
- `--scan` must match the scan.
- `--defocus`, `--rotation` and `--center` set the sharpness; tune them on a recorded scan
  (`tools/record_stream.py`, see the guide).
- `--detector-layout` handles a rotated or mirrored detector.
- `--accumulate latest` keeps the image updating while the microscope scans continuously.
- Results are saved in `out/`. Ctrl-C ends the run.

## Run without a detector

```bash
scripts/run_replay_demo.sh          # in Docker: replays a small sample scan
```

Without Docker or an NVIDIA GPU (CPU or Apple silicon):

```bash
pip install -e ".[local,viz,replay]"
scripts/run_local_demo.sh           # simulates a scan, replays it and shows it
```

The live image can also be watched from Jupyter: see
[`examples/live_view.ipynb`](examples/live_view.ipynb).

## Good to know

- After every scan the run prints `COMPLETE` or `WARNING: INCOMPLETE SCAN`.
- Only one program may read the DCU stream at a time.
- The DCU's stream compression must be `bslz4` (the default).
- More in [`docs/GUIDE.md`](docs/GUIDE.md), "Known issues and limitations".

## Tests

```bash
pip install -e ".[local,dev,oracles]"
python -m pytest tests/
```

## Credits

The real-time parallax / tcBF algorithm is the work of Georgios Varnavides. The GPU ingest in
`operators/` builds on [electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d)
(Denis Leshchev, NVIDIA; MIT license, see `operators/LICENSE`).
