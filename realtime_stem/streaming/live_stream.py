"""Live-preview transport shared by the producer (the Holoscan sink, in the container) and any
consumer (the example viewer/probe, on the host). Both import pack/unpack from HERE so the wire
format has exactly one definition.

Wire format — ONE frame (ZMQ CONFLATE does not support multipart):
    [4-byte big-endian header length][header JSON (utf-8)][raw C-contiguous array bytes]
    header = {"shape": [Ny, Nx], "dtype": "<f4", "seq": int, "final": bool}

PUB is non-blocking and lossy on purpose: publish() never stalls the hot path, and the
subscriber keeps only the newest frame (CONFLATE). No GPU / Holoscan dependency, so this module
and its tests run anywhere numpy + pyzmq are installed.
"""
from __future__ import annotations

import json
import struct
import time
from typing import Callable

import numpy as np
import zmq

_HLEN = struct.Struct(">I")   # 4-byte big-endian header-length prefix


def pack(arr: np.ndarray, *, seq: int, final: bool = False) -> bytes:
    """Serialize a numpy array + metadata into one frame."""
    a = np.ascontiguousarray(arr)
    header = json.dumps({
        "shape": [int(d) for d in a.shape],
        "dtype": a.dtype.str,          # e.g. "<f4" — round-trips width + endianness exactly
        "seq": int(seq),
        "final": bool(final),
    }).encode("utf-8")
    return _HLEN.pack(len(header)) + header + a.tobytes()


def unpack(frame: bytes) -> tuple[np.ndarray, dict]:
    """Inverse of pack(): (writable array copy, header dict)."""
    (hlen,) = _HLEN.unpack_from(frame, 0)
    header = json.loads(frame[4:4 + hlen].decode("utf-8"))
    arr = np.frombuffer(frame, dtype=np.dtype(header["dtype"]),
                        offset=4 + hlen).reshape(header["shape"])
    return np.array(arr), header       # copy: the frombuffer view is read-only over `frame`


class Throttle:
    """`ready()` returns True at most once per `interval` seconds; the first call returns True
    (emit an initial frame promptly). Clock injectable so tests don't sleep."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.perf_counter):
        self.interval = float(interval)
        self._clock = clock
        self._last: float | None = None

    def ready(self) -> bool:
        now = self._clock()
        if self._last is None or (now - self._last) >= self.interval:
            self._last = now
            return True
        return False


class LivePublisher:
    """Bind a PUB socket once; publish() sends one snapshot, non-blocking (drops if no peer)."""

    def __init__(self, endpoint: str, *, context: "zmq.Context | None" = None):
        self.endpoint = endpoint
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 2)      # tiny send queue; we only care about "latest"
        self._sock.setsockopt(zmq.LINGER, 0)      # never block on close
        self._sock.bind(endpoint)
        self._seq = 0

    def publish(self, arr: np.ndarray, *, final: bool = False) -> int:
        seq = self._seq
        try:
            self._sock.send(pack(arr, seq=seq, final=final), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass                                  # HWM hit with no consumer — drop, don't block
        self._seq += 1
        return seq

    def close(self) -> None:
        self._sock.close(linger=0)


class LiveSubscriber:
    """Connect a SUB socket with CONFLATE (keep only the newest frame). recv_latest() returns
    (array, header) or None on timeout so a GUI loop stays responsive."""

    def __init__(self, endpoint: str, *, context: "zmq.Context | None" = None):
        self.endpoint = endpoint
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)    # must be set before connect
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(endpoint)

    def recv_latest(self, timeout_ms: int = 100) -> "tuple[np.ndarray, dict] | None":
        if self._sock.poll(timeout_ms) & zmq.POLLIN:
            return unpack(self._sock.recv())
        return None

    def close(self) -> None:
        self._sock.close(linger=0)
