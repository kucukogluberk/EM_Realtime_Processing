"""Unit tests for the live-preview transport. Pure Python (numpy + pyzmq) — runs anywhere."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.streaming.live_stream import (
    LivePublisher,
    LiveSubscriber,
    Throttle,
    pack,
    unpack,
)


def test_pack_unpack_roundtrip_exact():
    arr = (np.arange(12, dtype=np.float32).reshape(3, 4) * 1.5) - 3.0
    out, header = unpack(pack(arr, seq=7, final=True))
    assert out.dtype == np.float32
    assert out.shape == (3, 4)
    assert np.array_equal(out, arr)
    assert header["seq"] == 7
    assert header["final"] is True
    assert header["shape"] == [3, 4]


def test_throttle_emits_at_most_once_per_interval():
    now = [0.0]
    thr = Throttle(0.2, clock=lambda: now[0])
    assert thr.ready() is True            # first call always emits
    now[0] = 0.1
    assert thr.ready() is False           # 0.1 s < 0.2 s
    now[0] = 0.2
    assert thr.ready() is True            # exactly the interval
    now[0] = 0.35
    assert thr.ready() is False
    now[0] = 0.41
    assert thr.ready() is True


def test_publish_without_subscriber_neither_blocks_nor_raises():
    pub = LivePublisher("tcp://127.0.0.1:5731")
    try:
        t0 = time.perf_counter()
        for _ in range(200):
            pub.publish(np.zeros((16, 16), dtype=np.float32))
        assert time.perf_counter() - t0 < 1.0     # non-blocking: 200 sends are ~instant
    finally:
        pub.close()


def test_pub_sub_delivers_latest_frame():
    endpoint = "tcp://127.0.0.1:5732"
    pub = LivePublisher(endpoint)
    sub = LiveSubscriber(endpoint)
    try:
        time.sleep(0.2)                            # SUB slow-joiner settle
        for i in range(5):
            pub.publish(np.full((4, 4), float(i), dtype=np.float32), final=(i == 4))
            time.sleep(0.02)
        got = None
        for _ in range(30):
            msg = sub.recv_latest(timeout_ms=100)
            if msg is not None:
                got = msg
                if msg[1]["final"]:
                    break
        assert got is not None, "subscriber received nothing"
        arr, header = got
        assert header["final"] is True
        assert np.all(arr == 4.0)                  # CONFLATE → newest frame (i=4)
    finally:
        pub.close()
        sub.close()
