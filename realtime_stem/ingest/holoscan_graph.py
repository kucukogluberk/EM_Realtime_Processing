"""The Holoscan ingest graph: ZmqRxOp → DecompressGpuOp → GenericSinkOp.

`ZmqRxOp` and `DecompressGpuOp` are this repository's C++/CUDA operators (`operators/`), compiled
into the image by `docker/Dockerfile` (imported as `dectris_holoscan_ops`): they receive raw ARINA messages over ZeroMQ and
bslz4-decompress them on the GPU, GIL-free, at full detector rate. The decompressor emits each
batch as CUDA tensors via DLPack — `frames` (B, H, W) uint16/uint32 and `image_ids` (B,) — so
`GenericSinkOp` is a zero-copy handoff into any `Accumulator`.

The graph is algorithm-agnostic: `GenericSinkOp` and `IngestApp` are built around the `Accumulator`
contract, so the same ingest drives parallax today and any future method unchanged.

Runs inside the container (this package is bind-mounted; the image supplies the compiled ops).
See `apps/run.py` for the command line and the README for the docker invocation.
"""

from __future__ import annotations

import time

import torch
from holoscan.core import Application, Operator, OperatorSpec

import dectris_holoscan_ops
from dectris_holoscan_ops import DecompressGpuOp, ZmqRxOp

from realtime_stem.processing.base import Accumulator

#: Version of the compiled operators in this image (None: built before 2.0, without the
#: end-of-scan and integrity fixes).
OPERATORS_VERSION: str | None = getattr(dectris_holoscan_ops, "__version__", None)


def _tensor_from_dlpack(tensor):
    """Convert a Holoscan Tensor-like object to a torch.Tensor without a copy."""
    if hasattr(tensor, "__dlpack__"):
        return torch.from_dlpack(tensor)
    if hasattr(tensor, "to_dlpack"):
        return torch.from_dlpack(tensor.to_dlpack())
    raise TypeError(f"Object does not expose DLPack: {type(tensor)!r}")


def _get_tensor(message, name: str):
    if isinstance(message, dict):
        return message[name]
    if hasattr(message, "get"):
        value = message.get(name)
        if value is not None:
            return value
    return message[name]


class GenericSinkOp(Operator):
    """Feed decompressed batches into any `Accumulator`; print throughput periodically; optionally
    PUB throttled live snapshots of the reconstruction-so-far."""

    def __init__(self, fragment, *args, acc: Accumulator, print_every: int = 50,
                 live_publisher=None, live_every: float = 0.2, name: str = "sink"):
        super().__init__(fragment, *args, name=name)
        self.acc = acc
        self.print_every = print_every
        self._start = None
        self._batches = 0
        self._frames = 0
        self._n_positions = acc.scan_shape[0] * acc.scan_shape[1]
        # Which scan positions have received a frame (on the GPU; one tiny scatter per batch).
        # The end-of-scan report counts missing positions and duplicate frames from it.
        self._seen = torch.zeros(self._n_positions, dtype=torch.bool, device="cuda")
        self.live_publisher = live_publisher
        # Lazy import so the default (no live) path never touches zmq.
        if live_publisher is not None:
            from realtime_stem.streaming.live_stream import Throttle
            self._throttle = Throttle(live_every)
        else:
            self._throttle = None

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        message = op_input.receive("input")
        frames = _tensor_from_dlpack(_get_tensor(message, "frames"))
        # ids stay on the device: a .cpu().numpy() here would cost a full D2H round trip per batch
        # (the engine would re-upload them) and cut throughput.
        ids = _tensor_from_dlpack(_get_tensor(message, "image_ids")).to(torch.long)

        if self._start is None:
            self._start = time.perf_counter()

        # Real ids from the stream (not assumed-sequential). Guard against a scan-shape
        # mismatch before they hit a GPU indexed op (device asserts are opaque). int() of a
        # device max is a scalar sync — far cheaper than shipping the whole array to host.
        #
        # That scalar sync is also the pipeline's only cross-stream fence: the accumulator's
        # torch work runs on torch's stream while DecompressGpuOp writes its 3-slot output
        # pool on its own CUDA stream, and this sync waits for the PREVIOUS batch's scatter
        # before the pool slot can be recycled under it. If you ever remove the guard, replace
        # it with an explicit torch.cuda.synchronize()/event.
        if ids.numel() and int(ids.max()) >= self._n_positions:
            raise ValueError(
                f"image_id {int(ids.max())} >= scan positions {self._n_positions} — "
                f"--scan doesn't match the acquisition."
            )
        self.acc.process(frames, ids)
        self._seen[ids] = True

        self._batches += 1
        self._frames += int(frames.shape[0])
        if self._batches % self.print_every == 0:
            elapsed = max(time.perf_counter() - self._start, 1e-9)
            print(f"[sink] batches={self._batches} frames={self._frames} "
                  f"fps={self._frames / elapsed:,.1f}", flush=True)

        # Live preview: throttled snapshot of the reconstruction-so-far (D2H sync gated to ~5 Hz).
        if self.live_publisher is not None and self._throttle.ready():
            self.live_publisher.publish(self.acc.result())

    def summary(self) -> str:
        if self._start is None:
            return "[sink] no batches received"
        elapsed = max(time.perf_counter() - self._start, 1e-9)
        return (f"[sink] TOTAL frames={self._frames} batches={self._batches} "
                f"elapsed={elapsed:.2f}s rate={self._frames / elapsed / 1e3:,.1f} kHz")

    def completeness(self) -> tuple[bool, str]:
        """(complete, one-line report): did every scan position receive exactly one frame?"""
        covered = int(self._seen.sum()) if self._start is not None else 0
        missing = self._n_positions - covered
        duplicates = self._frames - covered
        if missing == 0 and duplicates == 0:
            return True, f"[sink] COMPLETE: all {self._n_positions} scan positions received"
        parts = [f"{missing} missing"] if missing else []
        if duplicates:
            parts.append(f"{duplicates} frames for positions that already had one")
        return False, (f"[sink] WARNING: INCOMPLETE SCAN: {covered} of {self._n_positions} scan "
                       f"positions received ({', '.join(parts)}).")


class IngestApp(Application):
    """ZmqRxOp → DecompressGpuOp → GenericSinkOp, configured from the accumulator's geometry."""

    def __init__(self, endpoint: str, batch_size: int, scheduler_threads: int,
                 acc: Accumulator, dtype: str = "uint16", n_connections: int = 3,
                 live_publisher=None, live_every: float = 0.2):
        super().__init__()
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.scheduler_threads = scheduler_threads
        self.acc = acc
        self.dtype = dtype
        self.n_connections = n_connections
        self.live_publisher = live_publisher
        self.live_every = live_every
        self.sink: GenericSinkOp | None = None

    def compose(self) -> None:
        if self.scheduler_threads > 0:
            try:
                from holoscan.schedulers import MultiThreadScheduler
                # The grace period stops the scheduler from mistaking the instant between two
                # operators for a deadlock and ending the run early (seen on a busy machine).
                self.scheduler(MultiThreadScheduler(
                    self, worker_thread_number=self.scheduler_threads,
                    stop_on_deadlock_timeout=500))
            except Exception as exc:
                print(f"[holoscan] using default scheduler: {exc}", flush=True)

        # The generic C++ ops take detector geometry + dtype and validate them against the stream.
        # n_connections is a recent parameter (the DCU paces each TCP connection, so full rate
        # needs 3); if the installed ops predate it, construct without it. Ops without the
        # geometry parameters at all (the public live4d "basic" build: fixed 96x96 uint16) are
        # refused with a clear message rather than a bare TypeError.
        hy, hx = self.acc.detector_shape
        op_kwargs = {"detector_width": int(hx), "detector_height": int(hy),
                     "dtype": self.dtype}
        n_images = int(self.acc.scan_shape[0] * self.acc.scan_shape[1])
        try:
            try:
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", n_connections=int(self.n_connections),
                             expected_images=n_images, **op_kwargs)
            except TypeError:
                # Operators before 2.0 have no expected_images (the start-message check).
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", n_connections=int(self.n_connections), **op_kwargs)
            decompress = DecompressGpuOp(self, name="decompress_gpu", **op_kwargs)
        except TypeError:
            try:
                rx = ZmqRxOp(self, endpoint=self.endpoint, batch_size=self.batch_size,
                             name="zmq_rx", **op_kwargs)
                decompress = DecompressGpuOp(self, name="decompress_gpu", **op_kwargs)
            except TypeError as exc:
                raise RuntimeError(
                    "the installed dectris_holoscan_ops do not accept detector_width/"
                    "detector_height/dtype: this image carries the fixed 96x96 uint16 'basic' "
                    "operators (e.g. the public live4d build), not the generic ones this "
                    "pipeline needs. Build the image from this repository's docker/Dockerfile "
                    "(README, Prerequisites)."
                ) from exc
            if self.n_connections != 1:
                print("[holoscan] installed ops predate n_connections — using a single "
                      "connection (the DCU caps one connection below full rate; rebuild the "
                      "image from docker/Dockerfile for multi-connection)", flush=True)
        self.sink = GenericSinkOp(self, acc=self.acc, live_publisher=self.live_publisher,
                                  live_every=self.live_every, name="sink")

        self.add_flow(rx, decompress, {("output", "input")})
        self.add_flow(decompress, self.sink, {("output", "input")})
