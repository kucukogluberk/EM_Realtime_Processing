# Changelog

Each release is a git tag (`git checkout v2.0.2`). Build the image of a release from its own
checkout and tag it with the version, e.g. `docker build -t realtime-stem:v2.0.2 -f
docker/Dockerfile .`, so the versions can be switched back and forth.

## 2.0.2

- The default scheduler is Holoscan's single-threaded one (`--scheduler-threads 0`). The
  4-thread `MultiThreadScheduler` of earlier versions lets idle workers sleep up to 5 ms between
  checks for work, which costs 5–7 ms per batch: at a paced 120 kHz it ended a 1024² scan about
  0.5 s late (a backlog of about 60k frames) and fell further behind on a busy machine. The
  single-threaded scheduler kept up in every test and uses less CPU. `--scheduler-threads 4`
  restores the old behaviour. No change to the operators (still 2.0.1), the image or the
  reconstruction.
- Ctrl-C and `docker stop` end a `--continuous` session with either scheduler, also between two
  scans. Holoscan stops the running graph by itself on these signals and `app.run()` returns as
  if the scan had ended; the runner now records the signal instead of relying on a
  KeyboardInterrupt (which the single-threaded scheduler swallows, and which `docker stop` never
  raised: in 2.0.1 and earlier, `docker stop` of a `--continuous` session armed the next scan and
  was killed 10 s later, after the results were saved).
- `tools/record_stream.py` records a live scan to a `.dstream` file, message for message, for
  replay and for tuning the parameters offline (3 connections by default).

## 2.0.1

The reconstruction is unchanged.

- **Fix: about one frame per million was skipped.** The receiver finds a frame's compressed data
  by searching the message for the three bytes of the DECTRIS compression tag. The DCU's
  timestamps contain the same three bytes about once per million frames; the receiver then
  rejected that frame ("could not read a bslz4-compressed frame") and the scan was reported
  incomplete by one position. A match that is not followed by a valid payload is now skipped and
  the search goes on (found by checking a recorded DCU stream frame by frame; also in 1.0.0 and
  2.0.0). The
  operators are now 2.0.1.
- `--n-connections` defaults to 3. The DCU paces each connection to about 60k beam-on binned
  frames per second, so the former default of 2 capped a live scan near 112 kHz.
- `docker/Dockerfile` pins the base image (by digest) and the pip packages to the versions 2.0.0
  was tested with, so a later build gives the same image.
- `scripts/check_install.sh` also reports the speed of the network link towards the DCU (step 6).
- Docs: the GPU architectures compiled for; the network and connection requirements; the decode packages for
  `tools/make_dstream.py` are host-side (the image does not carry them).

## 2.0.0

Robustness at the edges of a scan. The reconstruction itself is unchanged: on the same data,
2.0.0 and 1.0.0 give the same image (float32 rounding).

**End of scan**

- The last, partial batch of a scan is now processed. Any batch size works, and scans whose size
  is not a multiple of the batch size (e.g. 500×500) are complete.
- After the `end` message the receiver keeps reading until every image the `start` message
  announced is in, or until the series has been quiet for 100 ms. With several connections, `end`
  can overtake frames still queued on another connection; 1.0.0 lost them (up to one batch,
  typically on the first scan after start-up).
- Only the receiver stops at the end of a scan; the decompressor and the reconstruction finish
  everything already received before the run ends.
- The scheduler gets a 500 ms grace period before it decides that the pipeline has nothing left
  to do. 1.0.0 could end a run in the middle of a scan, without an error, when the machine was
  busy.

**Scans in quick succession (`--continuous`)**

- The connection to the DCU stays open from one scan to the next. 1.0.0 closed it after every
  scan and threw away whatever had already arrived for the next scan, so a scan that started
  within about a second of the previous one lost its first frames. Messages of the next scan that
  arrive while the current one is still finishing are kept and processed by the next round.
  Tested with scans back to back at 120 kHz with no gap at all: every scan complete.

**Ctrl-C and stopping**

- The receiver waits for messages with a timeout, so Ctrl-C and `docker stop` work at any time,
  also while waiting for a scan or when a scan never sends its `end`. The image received so far is
  saved.

**Series handling**

- The receiver follows the stream's `series_id`. Frames of another series are never mixed into
  the current scan. A series that never sends its `end` is ended when the next series begins,
  with a warning.
- An `end` message left over from an earlier run (before any start or frame) is ignored instead
  of ending the run with nothing.
- Frames that arrive without their series' `start` (left over from a crashed run, or the
  pipeline was started during a scan) are reported with a warning, and the run reports the scan
  as incomplete.
- The `start` message is checked: a warning when the detector announces a different number of
  images than `--scan` has positions, or a different frame size than configured.

**Integrity**

- Every image's compressed data is checked on the GPU (number of LZ4 blocks, and nvcomp's status
  for every block). Corrupt data now stops the run with an error instead of giving a silently
  wrong image.
- Image messages that cannot be decoded (no bslz4 payload, no `image_id`) are skipped and
  counted; the first one is reported with a hint. 1.0.0 placed a frame without `image_id` at scan
  position 0.
- The message type is recognised wherever the `type` key sits in a small message.

**Reporting and results**

- After every scan the runner reports whether every scan position received exactly one frame:
  `COMPLETE`, or `WARNING: INCOMPLETE SCAN` with the number of missing positions and duplicate
  frames.
- `--continuous` also saves every scan as `parallax_recon_scanNNN.npy`; `parallax_recon.npy` is
  always the latest scan.
- If the run stops with an error (e.g. a frame id beyond `--scan`), the image so far is saved
  first.
- At start-up the runner prints its version and the operators' version, and warns when the DCU
  address does not accept connections.

## 1.0.0

First release: parallax / tcBF on the Holoscan ingest, live stream-out, replay tools, and the
C++/CUDA operators in `operators/`. The operators are those in use since July 2026, apart from the
batch ceiling (2048).
