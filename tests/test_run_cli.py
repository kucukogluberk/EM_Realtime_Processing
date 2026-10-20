"""Runner-level tests (no Holoscan, no GPU): `apps/run.py` imports holoscan lazily, so its CLI and
its pre-flight checks are testable anywhere the package imports."""

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.apps import run as run_mod
from realtime_stem.processing import registry


def test_parser_builds_without_holoscan():
    p = run_mod.build_parser(registry.get("parallax"))
    args = p.parse_args(["--algorithm", "parallax", "--binned", "--scan", "256", "256",
                         "--batch-size", "512", "--n-connections", "3"])
    assert args.batch_size == 512
    assert args.scan == [256, 256]
    assert args.binned and args.n_connections == 3


def test_default_connections_carry_full_rate():
    """The DCU paces each connection below 120 kHz (about 60k beam-on binned frames/s), so the
    default must be 3: two connections cap near 112 kHz and the DCU drops the rest."""
    args = run_mod.build_parser(registry.get("parallax")).parse_args(["--binned"])
    assert args.n_connections == 3


def test_default_scheduler_is_the_greedy_one():
    """The single-threaded scheduler keeps up with a paced 120 kHz stream (the 4-thread one fell
    about 0.5 s behind per 1024^2 scan), so it is the default since 2.0.2."""
    args = run_mod.build_parser(registry.get("parallax")).parse_args(["--binned"])
    assert args.scheduler_threads == 0


def test_endpoint_warning_flags_an_unreachable_endpoint():
    """The pre-flight check warns when nothing listens at the endpoint, stays silent when
    something does, and skips endpoints it should not check (local, or not tcp://host:port)."""
    host = _non_loopback_address()
    with socket.socket() as srv:
        srv.bind((host, 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert run_mod.endpoint_warning(f"tcp://{host}:{port}") is None
    w = run_mod.endpoint_warning(f"tcp://{host}:{port}")         # closed now
    assert w is not None and f"{host}:{port}" in w
    assert run_mod.endpoint_warning(f"tcp://127.0.0.1:{port}") is None   # local: not checked
    assert run_mod.endpoint_warning("ipc:///tmp/stream") is None
    assert run_mod.endpoint_warning("tcp://10.0.0.1") is None            # no port


def _non_loopback_address() -> str:
    """This machine's address on its default route (no packet is sent), else skip the test."""
    import pytest
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))           # TEST-NET-1: only picks the outgoing interface
            host = s.getsockname()[0]
        except OSError:
            host = ""
    if not host or host.startswith("127."):
        pytest.skip("no non-loopback IPv4 address")
    return host


def test_continuous_save_keeps_a_numbered_copy(tmp_path):
    """In continuous mode every scan also gets its own file, so a session keeps all its scans;
    the plain name always holds the latest one."""
    from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm
    algo = ParallaxAlgorithm()
    for k, tag in enumerate(("scan001", "scan002")):
        acc = SimpleNamespace(result=lambda k=k: np.full((4, 4), k, dtype=np.float32),
                              geometry=SimpleNamespace(shifts=np.zeros((3, 2), np.int32)))
        algo.save(acc, tmp_path, tag=tag)
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == ["parallax_recon.npy", "parallax_recon_scan001.npy",
                     "parallax_recon_scan002.npy", "parallax_shifts.npy"]
    assert np.load(tmp_path / "parallax_recon_scan001.npy").max() == 0
    assert np.load(tmp_path / "parallax_recon.npy").max() == 1          # the latest scan


def test_continuous_save_tags_every_channel(tmp_path):
    """With readout channels, the numbered copy is kept for each channel."""
    from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm
    acc = SimpleNamespace(images=lambda: {"raw": np.zeros((4, 4), np.float32),
                                          "phase_flipped": np.ones((4, 4), np.float32)},
                          geometry=SimpleNamespace(shifts=np.zeros((3, 2), np.int32)))
    ParallaxAlgorithm().save(acc, tmp_path, tag="scan001")
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == ["parallax_phase_flipped.npy", "parallax_phase_flipped_scan001.npy",
                     "parallax_recon.npy", "parallax_recon_scan001.npy", "parallax_shifts.npy"]


def _args(argv):
    algo = registry.get("parallax")
    return run_mod.build_parser(algo).parse_args(algo.preprocess_argv(argv))


def test_resolve_ingest():
    """--source is Python-only; auto takes Holoscan only where its modules import (not here)."""
    import importlib.util
    have_holoscan = all(importlib.util.find_spec(m) for m in run_mod._HOLOSCAN_MODULES)
    assert run_mod.resolve_ingest(_args(["--source", "x.dstream"])) == "python"
    assert run_mod.resolve_ingest(_args(["--ingest", "python"])) == "python"
    assert run_mod.resolve_ingest(_args(["--ingest", "holoscan"])) == "holoscan"
    assert run_mod.resolve_ingest(_args([])) == ("holoscan" if have_holoscan else "python")
    assert run_mod.resolve_ingest(_args(["--device", "mps"])) == "python"
    import pytest
    with pytest.raises(SystemExit):
        run_mod.resolve_ingest(_args(["--ingest", "holoscan", "--source", "x.npy"]))


def test_readout_flags_map_to_config():
    from realtime_stem.processing.parallax.config import readout_from_args
    cfg = readout_from_args(_args([]))
    assert cfg.phase_flip and cfg.dc_correct and not cfg.normalize and cfg.suppress_nyquist is None
    cfg = readout_from_args(_args(["--no-phase-flip", "--no-dc-correct", "--normalize",
                                   "--coverage", "--lowpass", "0.2", "--highpass", "0.01",
                                   "--suppress-nyquist", "off", "--band-limit"]))
    assert not cfg.phase_flip and not cfg.dc_correct and cfg.normalize and cfg.coverage_channel
    assert cfg.q_lowpass == 0.2 and cfg.q_highpass == 0.01
    assert cfg.suppress_nyquist is False and cfg.band_limit
