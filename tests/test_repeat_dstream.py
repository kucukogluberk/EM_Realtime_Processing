"""`tools/repeat_dstream.py`: a recorded scan looped into passes keeps every payload and the arrival
order, numbers the ids as the detector would (running on, or restarting per series), moves the
positions for a drifting specimen, and writes ids the receiver's C parser reads at any width."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]


def _capture(path, n=_NPOS, seed=0, order=None):
    from realtime_stem.ingest import codec
    from realtime_stem.ingest.dstream import write_magic, write_record

    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 1000, size=(n, 16, 16), dtype=np.uint16)
    with open(path, "wb") as fh:
        write_magic(fh)
        write_record(fh, codec.encode_control_message("start"))
        for i in (range(n) if order is None else order):
            write_record(fh, codec.encode_image_message(int(i), frames[i]))
        write_record(fh, codec.encode_control_message("end"))
    return frames


def _read(path):
    from realtime_stem.ingest.codec import decode_message
    from realtime_stem.ingest.dstream import iter_records

    msgs = [decode_message(raw) for raw in iter_records(path)]
    return [t for t, _, _ in msgs], [(i, f) for t, i, f in msgs if t == "image"]


def test_repeat_one_series_keeps_the_payloads_and_the_order(tmp_path):
    from tools.repeat_dstream import main

    order = np.random.default_rng(1).permutation(_NPOS)   # arrival order across connections
    frames = _capture(tmp_path / "s.dstream", order=order)
    main([str(tmp_path / "s.dstream"), str(tmp_path / "r.dstream"), "--passes", "3"])
    types, images = _read(tmp_path / "r.dstream")
    assert types[0] == "start" and types[-1] == "end" and types.count("start") == 1
    ids = [i for i, _ in images]
    assert ids == [k * _NPOS + int(i) for k in range(3) for i in order]
    for i, f in images:
        np.testing.assert_array_equal(f, frames[i % _NPOS])


def test_repeat_per_pass_series_and_drift(tmp_path):
    from tools.repeat_dstream import main

    frames = _capture(tmp_path / "s.dstream")
    main([str(tmp_path / "s.dstream"), str(tmp_path / "p.dstream"), "--passes", "2",
          "--series", "per-pass"])
    types, images = _read(tmp_path / "p.dstream")
    assert types.count("start") == 2 and types.count("end") == 2
    assert [i for i, _ in images] == list(range(_NPOS)) * 2
    main([str(tmp_path / "s.dstream"), str(tmp_path / "d.dstream"), "--passes", "3",
          "--scan", *map(str, _SCAN), "--drift", "1", "2"])
    _, images = _read(tmp_path / "d.dstream")
    ny, nx = _SCAN
    for i, f in images:
        k, p = divmod(i, _NPOS)
        r, c = divmod(p, nx)
        np.testing.assert_array_equal(f, frames[((r - k) % ny) * nx + (c - 2 * k) % nx])


def test_repeated_ids_parse_in_the_c_receiver(tmp_path):
    """Ids past 65535 and 2^32 change the CBOR integer's width; the receiver's parser
    (operators/cpp/common/fast_cbor_parse.c) must read them."""
    from realtime_stem.ingest import codec
    from tools.repeat_dstream import image_id, with_image_id

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    lib = tmp_path / "libfcp.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(lib),
                    str(REPO / "operators/cpp/common/fast_cbor_parse.c")], check=True)
    fcp = ctypes.CDLL(str(lib))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fcp.fast_extract_compressed_bslz4.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint64)]
    msg = codec.encode_image_message(7, np.arange(256, dtype=np.uint16).reshape(16, 16))
    for new in (7, 300, 70000, 5 * 2**32 + 3):
        patched = with_image_id(msg, new)
        assert image_id(patched) == new and codec.decode_message(patched)[1] == new
        ptr, size, orig, elem, iid = (u8p(), ctypes.c_size_t(), ctypes.c_size_t(),
                                      ctypes.c_size_t(), ctypes.c_uint64())
        rc = fcp.fast_extract_compressed_bslz4(patched, len(patched), ctypes.byref(ptr),
                                               ctypes.byref(size), ctypes.byref(orig),
                                               ctypes.byref(elem), ctypes.byref(iid))
        assert rc == 0 and iid.value == new and orig.value == 512
