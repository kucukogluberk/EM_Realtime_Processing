"""Tests for live OBF (`realtime_stem/processing/obf`).

The port itself is checked against quantem offline, on identical inputs, to 2e-6 relative on the
Gd2O3 recordings (`sessions/2026-10-20_Gd2O3/analysis/obf_vs_quantem.py`). Here the oracle is
`reconstruct_once`, the whole store in one call, and the physics a simulated weak phase object:

  • geometry: `bf_offsets` inverts `preprocess_geometry`'s placement, and a column-flipped detector
    gives the same offsets and sub-pixel centre as its canonical layout;
  • the numerator is linear and order-free: deposits in any chunks and order give one image, and
    `latest` replacements (new − old) give the image of the frames held;
  • the follower thread, fed in batches with revisits and out of order, ends exactly on
    `reconstruct_once` of the final store; a retune rebuilds it; off and on again;
  • a simulated weak phase object comes out with the right sign and closer to the truth than the
    phase-flipped parallax image, on both placements (lattice and patch).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_stem.ingest.sink import ACC_LOCK  # noqa: E402
from realtime_stem.processing.obf import core  # noqa: E402
from realtime_stem.processing.obf.core import (  # noqa: E402
    ObfSetup,
    bf_offsets,
    center_fraction,
    normalize,
    reconstruct_once,
)
from realtime_stem.processing.obf.follower import _SETUPS, ObfFollower  # noqa: E402
from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402
from realtime_stem.processing.parallax.readout import fourier_filters  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (
    ["cuda"] if torch.cuda.is_available() else [])

_SCAN = (12, 10)
_NPOS = _SCAN[0] * _SCAN[1]


def _geometry(c10=-1.5e4, rotation=-15.0, upsampling=2, center=(16.3, 15.6), flip=True, scan=_SCAN):
    return preprocess_geometry(
        (32, 32), scan, (256 / 12, 256 / 12), 300e3, 4.0, (0.307617, 0.307617),
        aberration_coefs={"C10": c10, "C12": 300.0, "phi12": 0.4},
        rotation_angle=math.radians(rotation), upsampling_factor=upsampling,
        probe_center=center, flip_cols=flip)


def _frames(n, seed=0):
    return np.random.default_rng(seed).integers(800, 1200, size=(n, 32, 32), dtype=np.uint16)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _tol(device):
    return 1e-9 if device == "cpu" else 2e-5


def _dtype(device):
    return torch.float64 if device == "cpu" else None


@pytest.fixture(autouse=True)
def _fresh_setups():
    _SETUPS.clear()
    yield
    _SETUPS.clear()


# ── geometry ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("flip", [False, True])
def test_bf_offsets_invert_the_placement(flip):
    geo = _geometry(center=(15.7, 17.2), flip=flip)
    off = bf_offsets(geo)
    canonical = _geometry(center=(15.7, 17.2) if not flip else (15.7, 31 - 17.2), flip=False)
    np.testing.assert_array_equal(off, bf_offsets(canonical))
    np.testing.assert_allclose(center_fraction(geo), center_fraction(canonical))
    # every offset is inside the semiangle, and offset 0 is the rounded centre's pixel
    assert (np.hypot(*off.T) <= 4.0 / 0.307617 + 1e-9).all()
    np.testing.assert_allclose(center_fraction(_geometry(center=(15.7, 17.2), flip=False)),
                               [0.3, -0.2], atol=1e-12)


# ── the numerator ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("device", DEVICES)
def test_deposit_is_linear_and_order_free(device):
    geo = _geometry()
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    rng = np.random.default_rng(1)
    values = torch.as_tensor(rng.standard_normal((_NPOS, geo.n_bf))).to(setup.rdtype).to(device)
    pos = torch.arange(_NPOS, device=device)
    whole = setup.buffer()
    setup.deposit(whole, values, pos)
    parts = setup.buffer()
    order = torch.as_tensor(rng.permutation(_NPOS), device=device)
    for chunk in order.split(17):
        setup.deposit(parts, values[chunk], pos[chunk])
    a, b = setup.image(whole).cpu().numpy(), setup.image(parts).cpu().numpy()
    assert _rel(b, a) < 10 * _tol(device)
    doubled = setup.buffer()
    setup.deposit(doubled, 2 * values, pos)
    assert _rel(setup.image(doubled).cpu().numpy(), 2 * a) < 10 * _tol(device)


@pytest.mark.parametrize("device", DEVICES)
def test_latest_replacements_are_the_frames_held(device):
    geo = _geometry()
    setup = ObfSetup.build(geo, device=device, dtype=_dtype(device))
    rdtype = setup.rdtype
    first = torch.as_tensor(_frames(_NPOS, 0).reshape(_NPOS, -1)[:, geo.bf_flat_inds]).to(rdtype)
    second = torch.as_tensor(_frames(_NPOS, 1).reshape(_NPOS, -1)[:, geo.bf_flat_inds]).to(rdtype)
    first, second = first.to(device), second.to(device)
    mean = first.mean(0)
    pos = torch.arange(_NPOS, device=device)
    replaced = torch.as_tensor(np.random.default_rng(2).permutation(_NPOS)[:50], device=device)
    buffer = setup.buffer()
    setup.deposit(buffer, normalize(first, mean), pos)
    setup.deposit(buffer, normalize(second[replaced], mean) - normalize(first[replaced], mean),
                  replaced)
    held = first.clone()
    held[replaced] = second[replaced]
    fresh = setup.buffer()
    setup.deposit(fresh, normalize(held, mean), pos)
    assert _rel(setup.image(buffer).cpu().numpy(), setup.image(fresh).cpu().numpy()) < \
        10 * _tol(device)


# ── the follower thread ────────────────────────────────────────────────────────

def _feed(acc, passes, seed=0, batch=32, shuffle=True):
    """Passes of frames in batches, under ACC_LOCK as the sink feeds them; ids out of order
    within a pass when `shuffle`."""
    rng = np.random.default_rng(seed)
    for p in range(passes):
        ids = rng.permutation(_NPOS) if shuffle else np.arange(_NPOS)
        frames = _frames(_NPOS, seed + p)
        for lo in range(0, _NPOS, batch):
            sel = ids[lo:lo + batch]
            with ACC_LOCK:
                acc.process(frames[sel], sel + p * _NPOS)


def _expected(acc, geo):
    """`reconstruct_once` of the store, through the readout's Nyquist filter as the follower."""
    with ACC_LOCK:
        img = reconstruct_once(acc._store, acc._visits, acc._store_kind, geo).cpu().numpy()
    base, _ = fourier_filters(geo.upsampled_scan_gpts, geo.upsampled_scan_sampling,
                              geo.wavelength, geo.aberration_coefs,
                              suppress_nyquist=geo.upsampling_factor > 1)
    return img if base is None else np.fft.ifft2(np.fft.fft2(img) * base).real


@pytest.mark.parametrize("device", DEVICES)
def test_follower_ends_on_the_frames_held(device):
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device=device, accumulate="latest", wrap_ids=True)
    follower = ObfFollower(acc, chunk=24, live_every=0.02)
    try:
        _feed(acc, passes=1)
        with ACC_LOCK:                         # a partial second pass: some positions replaced
            acc.process(_frames(40, 7), np.arange(40) + 3 * _NPOS // 2)
        image = follower.finalize(timeout=60)
        assert image is not None and follower.state == "live"
        assert _rel(image, _expected(acc, geo)) < 1e-4
        with ACC_LOCK:
            stack, names, meta = acc.snapshot()
        assert names[-1] == "obf" and stack.shape == (len(names), *geo.upsampled_scan_gpts)
        assert meta["obf_state"] == "live" and meta["obf_coverage"] == 1.0
        with ACC_LOCK:
            assert "obf" in acc.images()
    finally:
        acc.close()


def test_follower_starts_on_a_store_that_is_already_full():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    _feed(acc, passes=2, shuffle=False)
    follower = ObfFollower(acc, live_every=0.02)
    try:
        assert _rel(follower.finalize(timeout=60), _expected(acc, geo)) < 1e-4
    finally:
        acc.close()


def test_follower_follows_a_retune_and_can_be_switched_off():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    follower = ObfFollower(acc, live_every=0.02)
    try:
        _feed(acc, passes=1)
        before = follower.finalize(timeout=60)
        retuned = _geometry(c10=-1.3e4, rotation=-14.0)
        with ACC_LOCK:
            acc.set_geometry(retuned)
        after = follower.finalize(timeout=60)
        assert _rel(after, _expected(acc, retuned)) < 1e-4
        assert _rel(after, before) > 0.05             # the geometry changed the image
        with ACC_LOCK:
            follower.set_enabled(False)
        deadline = time.monotonic() + 10
        while follower.state != "off" and time.monotonic() < deadline:
            time.sleep(0.02)
        with ACC_LOCK:
            _, names, meta = acc.snapshot()
        assert "obf" not in names and meta["obf_state"] == "off" and acc._obf_dirty is None
        with ACC_LOCK:
            follower.set_enabled(True)                # the store is swept again from the start
        assert _rel(follower.finalize(timeout=60), _expected(acc, retuned)) < 1e-4
    finally:
        acc.close()


def test_the_channel_is_obf_where_covered_and_the_phase_flipped_stand_in_elsewhere():
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    follower = ObfFollower(acc, enabled=False)                 # the thread idles
    try:
        rng = np.random.default_rng(3)
        ny, nx = geo.upsampled_scan_gpts
        image = rng.standard_normal((ny, nx)).astype(np.float32) * 0.1
        standin = rng.standard_normal((ny, nx)).astype(np.float32) * 500 + 3000
        assert follower.compose(standin)[0] is standin or \
            np.array_equal(follower.compose(standin)[0], standin)   # nothing yet: the stand-in
        done = np.zeros(ny, bool)
        done[6:15] = True
        follower.image_np, follower.done_np = image, done
        out, rows = follower.compose(standin)
        assert rows == [6, 15]
        np.testing.assert_array_equal(out[done], image[done])
        scale = image[done].std() / standin[done].std()
        expected = scale * (standin - standin[done].mean()) + image[done].mean()
        np.testing.assert_allclose(out[~done], expected[~done], rtol=1e-4, atol=1e-6)
        follower.done_np = None                                  # every frame covered: OBF alone
        assert follower.compose(standin) == (image, None) or \
            np.array_equal(follower.compose(standin)[0], image)
    finally:
        acc.close()


def test_an_output_row_shows_obf_once_the_frames_within_the_probe_are_covered():
    geo = _geometry()                                         # ×2, 12 x 10 scan
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    follower = ObfFollower(acc, enabled=False)
    try:
        u = geo.upsampling_factor
        reach = int(np.ceil(np.hypot(*np.asarray(geo.shifts, float).T).max() / u)) + 2
        follower._seen[:] = True
        follower._seen.reshape(_SCAN)[0, 3] = False            # one frame of scan row 0 missing
        done = follower._done_rows()
        scan_row = np.arange(_SCAN[0] * u) // u
        np.testing.assert_array_equal(done, scan_row > reach)
    finally:
        acc.close()


def test_a_slow_setup_never_holds_the_ingest(monkeypatch):
    """The Gram is built on the follower's thread: batches keep their pace meanwhile."""
    built = ObfSetup.build

    def slow(*args, **kwargs):
        time.sleep(1.5)
        return built(*args, **kwargs)

    monkeypatch.setattr(core.ObfSetup, "build", slow)
    geo = _geometry()
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    follower = ObfFollower(acc, live_every=0.02)
    try:
        frames = _frames(_NPOS)
        worst = 0.0
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            t0 = time.perf_counter()
            with ACC_LOCK:
                acc.process(frames[:30], np.arange(30))
            worst = max(worst, time.perf_counter() - t0)
        assert follower.state == "building" and worst < 0.2
        assert follower.finalize(timeout=60) is not None
    finally:
        acc.close()


def test_runner_saves_obf(tmp_path):
    """run.py --compute-obf on a replayed capture (a pass and a bit): parallax_obf.npy is the OBF of the
    frames held at the end, and the live stream carried the channel."""
    import subprocess

    frames = _frames(_NPOS + 30, seed=9)
    np.save(tmp_path / "scan.npy", frames)
    flags = ["--detector", "32", "32", "--scan", *map(str, _SCAN), "--energy", "300e3",
             "--semiangle", "4", "--angular-sampling", "0.307617", "--scan-sampling",
             f"{256 / 12}", "--defocus=-1.5e4", "--rotation=-15", "--c12=300",
             f"--phi12={math.degrees(0.4)}", "--center", "16.3", "15.6", "--flip-cols",
             "--upsampling", "2"]
    out = subprocess.run(
        [sys.executable, "-m", "realtime_stem.apps.run", "--device", "cpu", "--out-dir",
         str(tmp_path / "out"), *flags, "--source", str(tmp_path / "scan.npy"), "--batch-size",
         "40", "--accumulate", "latest", "--compute-obf"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "finishing the image of every frame held" in out.stdout
    acc = ParallaxAccumulator(_geometry(), device="cpu", accumulate="latest", wrap_ids=True)
    acc.process(frames[:_NPOS], np.arange(_NPOS))
    acc.process(frames[_NPOS:], np.arange(_NPOS, _NPOS + 30))
    saved = np.load(tmp_path / "out" / "parallax_obf.npy")
    assert _rel(saved, _expected(acc, _geometry())) < 1e-4


# ── physics: a simulated weak phase object ─────────────────────────────────────

@pytest.mark.parametrize("scan, placement", [((48, 48), "lattice"), ((64, 64), "patch")])
def test_weak_phase_object(scan, placement):
    """Noiseless frames of a weak phase object: OBF has the sign of the phase and is closer to the
    (band-limited) truth than the phase-flipped parallax image."""
    from tools.simulate_4dstem import SimParams, Simulation

    p = SimParams(scan=scan, detector=(48, 48), semiangle=10.0, angular_sampling=0.5,
                  scan_step=2.0, aberrations={"C10": -2000.0}, rotation_deg=-15.0, dose=0,
                  n_particles=6, seed=3)
    sim = Simulation(p, device="cpu")
    frames = np.concatenate([f for _, f in sim.batches(512)])
    g = sim.grid
    geo = preprocess_geometry((48, 48), scan, (g.scan_step,) * 2, p.energy, p.semiangle,
                              (p.angular_sampling,) * 2, aberration_coefs={"C10": -2000.0},
                              rotation_angle=math.radians(-15.0))
    acc = ParallaxAccumulator(geo, device="cpu", accumulate="latest", wrap_ids=True)
    acc.process(frames, np.arange(len(frames)))
    setup = ObfSetup.build(geo, device="cpu", dtype=torch.float64)
    assert setup.placement == placement
    obf = reconstruct_once(acc._store, acc._visits, acc._store_kind, geo, setup=setup).numpy()
    flipped = acc.channels()["phase_flipped"].numpy()
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(scan[0], g.scan_step),
                              np.fft.fftfreq(scan[1], g.scan_step), indexing="ij"))
    band = q < 2 * p.semiangle * 1e-3 / g.wavelength
    truth = np.fft.ifft2(np.fft.fft2(sim.scan_phase()) * band).real

    def corr(a):
        return np.corrcoef(a.ravel(), truth.ravel())[0, 1]

    assert corr(obf) > 0.7
    assert corr(obf) > corr(flipped) + 0.05
