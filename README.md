# realtime-stem

Real-time STEM processing at full detector frame rate. Frames stream from a DECTRIS ARINA detector
(through its detector control unit, the DCU), are decompressed on an NVIDIA GPU and folded into a
reconstruction **while the scan is running**, so you watch the image fill in during the scan.

The included method is **parallax / tilt-corrected bright field (tcBF)**. It runs at 120 kHz on
binned (96×96) ARINA frames.

- [`docs/GUIDE.md`](docs/GUIDE.md): the full technical guide
- [`CHANGELOG.md`](CHANGELOG.md): what changed between versions
- [`AGENTS.md`](AGENTS.md): orientation for AI coding assistants

## What is where

```
realtime_stem/        the Python package
  apps/run.py           the program you run
  ingest/               receives the detector frames
  processing/parallax/  the parallax / tcBF reconstruction
  streaming/            publishes the live image
operators/            the C++/CUDA frame receiver and GPU decompressor
docker/Dockerfile     builds the Docker image
examples/             live viewer, result plotting, sample recording
tools/                stream recorder, replay
scripts/              installation check and demo
tests/                unit tests
```

## Requirements

- Linux with an NVIDIA GPU, Docker and the NVIDIA container toolkit (`docker run --gpus all`
  must work)
- Python 3.10+ on the host, for the viewer

## Install

```bash
docker build -t realtime-stem:latest -f docker/Dockerfile .
pip install -e ".[viz,replay]"
scripts/check_install.sh
```

The image holds the compiled code; the Python package is mounted from this folder at run time.

## Run

Start the pipeline before the scan:

```bash
docker run --rm --network host --gpus all --ulimit stack=33554432:33554432 \
  -v "$PWD":/work -v "$PWD/out":/out -w /work --entrypoint python3 \
  realtime-stem:latest -m realtime_stem.apps.run \
    --endpoint tcp://<DCU_IP>:31001 --binned --scan 256 256 \
    --defocus <Å> --rotation <degrees> \
    --live-endpoint tcp://127.0.0.1:5601 --continuous --out-dir /out
```

Watch it in a second terminal:

```bash
python examples/viewer.py tcp://127.0.0.1:5601
```

- `--scan` must match the scan.
- `--defocus`, `--rotation` and `--center` set the sharpness; tune them on a recorded scan
  (`tools/record_stream.py`, see the guide).
- Results are saved in `out/`. Ctrl-C ends the run.
- All options: add `--help` to the `run` command.

## Run without a detector

```bash
scripts/run_replay_demo.sh          # replays a small sample scan
```

## Good to know

- After every scan the run prints `COMPLETE` or `WARNING: INCOMPLETE SCAN`.
- Only one program may read the DCU stream at a time.
- The DCU's stream compression must be `bslz4` (the default).
- More in [`docs/GUIDE.md`](docs/GUIDE.md), "Known issues and limitations".

## Tests

```bash
python -m pytest tests/
```

## Credits

The real-time parallax / tcBF algorithm is the work of Georgios Varnavides. The GPU ingest in
`operators/` builds on [electronmicroscopy/live4d](https://github.com/electronmicroscopy/live4d)
(Denis Leshchev, NVIDIA; MIT license, see `operators/LICENSE`).
