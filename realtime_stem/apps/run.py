"""Run a real-time reconstruction on the Holoscan ingest.

    python -m realtime_stem.apps.run --algorithm parallax [transport flags] [algorithm flags]

This is the one place ingest → processing → stream-out are wired together. It is deliberately
generic: it owns the transport flags (endpoint, batching, scheduler, live preview) and delegates
everything method-specific to the selected `Algorithm` (`--algorithm`). Adding a method needs no
change here — see `realtime_stem/processing/registry.py`.

Runs inside the image built by `docker/Dockerfile` (this package bind-mounted), e.g.:

    docker run --rm --network host --gpus all --ulimit stack=33554432:33554432 \
      -v "$PWD":/work -v "$PWD/out":/out --entrypoint python3 \
      realtime-stem:latest -m realtime_stem.apps.run \
      --algorithm parallax --endpoint tcp://<DCU_IP>:31001 --batch-size 2048 \
      --binned --scan 256 256 --out-dir /out

`--continuous` (optionally `--rounds N`) re-arms with a fresh accumulator after each scan's `end`,
so a whole session of scans runs against one container. See the README for the full walkthrough
and the offline replay demo (no detector required).

After every scan the runner reports whether each scan position received exactly one frame
("COMPLETE" or "WARNING: INCOMPLETE SCAN"); the result is saved either way.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Make the package importable when run as a bare script from a bind mount
# (python3 /work/realtime_stem/apps/run.py): parents[2] is the repo root that holds the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from realtime_stem import __version__   # noqa: E402
from realtime_stem.processing import registry   # noqa: E402
from realtime_stem.processing.base import Algorithm   # noqa: E402


def endpoint_warning(endpoint: str, timeout: float = 2.0) -> str | None:
    """A warning when nothing accepts TCP connections at a `tcp://host:port` endpoint, else None.

    ZeroMQ connects lazily and retries forever, so a wrong DCU address, a disabled stream or a
    firewall otherwise looks exactly like "waiting for a scan". This opens and immediately closes
    one plain TCP connection: it carries no ZeroMQ handshake, so the sender never treats it as a
    receiver and no frame goes to it. Purely advisory: the pipeline starts either way. Local
    endpoints are not checked (a replay feeder there usually starts after the pipeline).
    """
    parts = urlsplit(endpoint)
    if parts.scheme != "tcp" or not parts.hostname or parts.port is None:
        return None
    if parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return None
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout=timeout):
            return None
    except OSError as exc:
        return (f"[run] WARNING: cannot reach {parts.hostname}:{parts.port} ({exc}). Check the "
                f"DCU address and port, that its stream interface is enabled, and the network / "
                f"firewall. The pipeline keeps waiting and connects once it is reachable.")


def build_parser(algorithm: Algorithm) -> argparse.ArgumentParser:
    """The full CLI: generic transport flags + the selected algorithm's own flags."""
    p = argparse.ArgumentParser(
        description="Real-time STEM reconstruction on the Holoscan ingest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algorithm", default=registry.names()[0], choices=registry.names(),
                   help="Which reconstruction algorithm to run.")

    # ── transport (algorithm-agnostic) ────────────────────────────────────────
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555",
                   help="ZMQ PULL endpoint (DCU: tcp://<dcu-ip>:31001; replay demo: localhost).")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Frames per ingest batch (any size works; the last, partial batch of a "
                        "scan is processed too). The compiled ingest caps this (kMaxBatchSize in "
                        "the operators' constants.h; larger values are clamped, with a warning "
                        "in the log).")
    p.add_argument("--scheduler-threads", type=int, default=0,
                   help="0 = Holoscan's single-threaded (greedy) scheduler, the fastest here; N > 0 "
                        "= MultiThreadScheduler with N workers, whose idle workers sleep up to 5 ms "
                        "between checks for work (5-7 ms per batch).")
    p.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16",
                   help="Pixel dtype of the stream (ARINA can send uint32 in long-dwell modes).")
    p.add_argument("--n-connections", type=int, default=3,
                   help="TCP connections to the stream. The DCU paces each connection (about "
                        "60k beam-on binned frames/s each), so 120 kHz needs 3; larger "
                        "(unbinned) frames can need more.")
    p.add_argument("--out-dir", type=Path, default=Path.cwd(),
                   help="Where results land.")
    p.add_argument("--live-endpoint", default="",
                   help="If set (e.g. tcp://127.0.0.1:5601), PUB throttled live snapshots of the "
                        "reconstruction for a subscriber. Empty (default) = no live publishing.")
    p.add_argument("--live-every", type=float, default=0.2,
                   help="Minimum seconds between live snapshots (0.2 = ~5 Hz).")
    p.add_argument("--continuous", action="store_true",
                   help="Don't exit after one stream: on the series 'end', save results, reset the "
                        "accumulator, and re-arm for the next scan (Ctrl-C / docker rm to stop). "
                        "Every scan must use the SAME geometry.")
    p.add_argument("--rounds", type=int, default=0, metavar="N",
                   help="With --continuous (implied if set): exit cleanly after N streams "
                        "(default 0 = keep going forever).")

    # ── algorithm-specific (geometry, physics, tuning knobs) ──────────────────
    algorithm.add_cli_args(p)
    return p


def finish_scan(app, acc, algorithm: Algorithm, out_dir: Path, publisher, tag: str) -> None:
    """Report one scan (frame count, completeness), save its results, land the viewer on them."""
    if app.sink is not None:
        print(app.sink.summary(), flush=True)
        print(app.sink.completeness()[1], flush=True)
    algorithm.save(acc, out_dir, tag=tag)
    if publisher is not None:
        publisher.publish(acc.result(), final=True)   # land the viewer on the final frame


def main() -> None:
    argv = sys.argv[1:]

    # Peek --algorithm so the right algorithm can add its flags to the real parser.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--algorithm", default=registry.names()[0], choices=registry.names())
    known, _ = peek.parse_known_args(argv)
    algorithm = registry.get(known.algorithm)

    parser = build_parser(algorithm)
    args = parser.parse_args(algorithm.preprocess_argv(argv))

    # holoscan lives only in the container — import lazily so --help / config / tests need no it.
    from realtime_stem.ingest.holoscan_graph import OPERATORS_VERSION, IngestApp

    print(f"[run] realtime-stem {__version__}, operators {OPERATORS_VERSION or 'before 2.0'}",
          flush=True)
    if OPERATORS_VERSION is None:
        print("[run] WARNING: this image's operators predate 2.0: no end-of-scan drain, no "
              "integrity check, Ctrl-C may not work while idle. Rebuild the image from "
              "docker/Dockerfile.", flush=True)
    warning = endpoint_warning(args.endpoint)
    if warning:
        print(warning, flush=True)

    publisher = None
    if args.live_endpoint:
        from realtime_stem.streaming.live_stream import LivePublisher   # lazy: only if live is on
        publisher = LivePublisher(args.live_endpoint)
        print(f"[run] live preview PUB on {args.live_endpoint} (every {args.live_every}s)",
              flush=True)

    # Ctrl-C (SIGINT) and `docker stop` (SIGTERM): Holoscan stops the running graph gracefully by
    # itself, and app.run() then returns as if the scan had ended. This flag tells the loop that
    # the run was stopped. (A KeyboardInterrupt is not enough: with the single-threaded scheduler
    # it is raised inside an operator and swallowed there, and SIGTERM raises nothing.)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    # One iteration = one stream (the ingest graph shuts down at the series 'end'). Continuous
    # mode re-arms with a FRESH accumulator and a fresh app: Holoscan Applications are one-shot,
    # but building both takes well under a second — far shorter than the detector re-arm.
    continuous = args.continuous or args.rounds > 0
    rounds_done = 0
    try:
        while True:
            rounds_done += 1
            acc = algorithm.build(args)   # builds geometry + accumulator, prints its own summary
            if rounds_done == 1:
                scheduler = ("single-threaded" if args.scheduler_threads <= 0
                             else f"{args.scheduler_threads} threads")
                print(f"[run] listening on {args.endpoint}, scan {acc.scan_shape}, "
                      f"detector {acc.detector_shape} {args.dtype}, batch {args.batch_size}, "
                      f"connections {args.n_connections}, scheduler {scheduler}", flush=True)
            app = IngestApp(args.endpoint, args.batch_size, args.scheduler_threads, acc,
                            dtype=args.dtype, n_connections=args.n_connections,
                            live_publisher=publisher, live_every=args.live_every)
            if continuous:
                total = f"/{args.rounds}" if args.rounds else ""
                print(f"[run] ── round {rounds_done}{total}: armed, waiting for a scan …",
                      flush=True)
            # In continuous mode every scan also gets its own numbered result file.
            tag = f"scan{rounds_done:03d}" if continuous else ""
            if stop["requested"]:      # stopped between two scans: the last one is saved already
                print("[run] stopped.", flush=True)
                break
            try:
                app.run()          # returns after the stream's `end` message, or on a stop signal
            except KeyboardInterrupt:
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                break
            except Exception:
                # e.g. a frame id beyond --scan: keep what was reconstructed, then fail loudly.
                print("[run] the pipeline stopped with an error — saving partial results.",
                      flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                raise
            if stop["requested"]:
                print("[run] interrupted — saving partial results.", flush=True)
                finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
                break
            finish_scan(app, acc, algorithm, args.out_dir, publisher, tag)
            if not continuous or (args.rounds and rounds_done >= args.rounds):
                break
    finally:
        if publisher is not None:
            publisher.close()


if __name__ == "__main__":
    main()
