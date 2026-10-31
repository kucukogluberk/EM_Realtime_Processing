"""`ObfFollower`: OBF as a live channel of the parallax accumulator, computed off the ingest thread.

OBF costs about a hundred times more per frame than parallax (`scripts/bench_obf.py`), so it never
runs inside `process()`, and it runs only while it is shown. Its weights (the Gram) depend on the
geometry and not on the data, and can be kept built for the current geometry meanwhile
(`precompute`), so that showing OBF needs no more than the frames' own work.

When OBF is switched on, or after a retune, its numerator starts empty and covers the scan from the
scan line outwards:

- forward with the scan: each new frame is deposited as it arrives (the ingest only marks the
  positions it replaced, `ParallaxAccumulator._obf_dirty`, one small scatter per batch);
- backward through the frames the `latest` store already holds, newest first, with the time the
  worker has left;
- a covered position the scan visits again is updated, its new values minus those deposited before
  (kept in a shadow of the store), so the numerator is always exactly the sum over the frames it
  covers, whatever order they came in.

Until every position held is covered the channel is a composite (`compose`): OBF on the output rows
it covers, and elsewhere the phase-flipped parallax image of the same frames, scaled to OBF's
contrast. So choosing OBF shows an image at once, and OBF wipes in from the scan line. Slower than
the frames, the worker lags: positions replaced again before it reaches them are deposited once, at
their newest frame, and the ingest never waits.

- On CUDA the worker's device work runs on its own stream. MPS's command queue takes one thread at a
  time, so there the worker holds `ACC_LOCK` for each unit of device work (one deposit sub-batch,
  ~20 ms binned) and builds the Gram on the CPU: the laptop is a development target.
- The normalization V = I/m − 1 needs each bright-field pixel's mean m. It is taken over the frames
  held when the coverage starts and frozen. A new m restarts the coverage: while fewer than
  `MIN_MEAN_POSITIONS` are held, each time they double, and when m drifts by more than
  `renorm_drift` rms. The final image (`finalize`) is re-swept with the exact means.

States: `off`, `building` (the weights), `filling` (the coverage growing), `live` (every position
held covered), `error`.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch

from realtime_stem.ingest.sink import ACC_LOCK
from realtime_stem.processing.obf.core import (
    ObfSetup,
    _raw_float,
    geometry_key,
    normalize,
)
from realtime_stem.processing.parallax.readout import fourier_filters

__all__ = ["ObfFollower"]

_SETUPS: "OrderedDict[tuple, ObfSetup]" = OrderedDict()
_SETUPS_LOCK = threading.Lock()
SETUP_CACHE_SIZE = 3            # setups kept (each holds a Gram and the operands, ~0.2 GB binned)
#: positions the frozen bright-field means are taken over at least (while fewer are held, they are
#: renewed each time the positions held double). Means from the first 8 rows of the binned Gd2O3
#: scan, 1.6% rms off the scan's, change its OBF image by 1.1% (correlation 0.99995)
MIN_MEAN_POSITIONS = 2048
#: scan rows beyond the probe's radius that an output row still waits for before it shows OBF
_REACH_MARGIN = 2


@contextlib.contextmanager
def _turn():
    """One unit of the worker's MPS work under ACC_LOCK, then a pause: a lock is not fair, and
    without it the worker would take it straight back from a waiting ingest."""
    with ACC_LOCK:
        yield
    time.sleep(5e-4)


@contextlib.contextmanager
def _hold():
    """The worker's hold of ACC_LOCK to read the store, then a pause: a lock is not fair, and
    without it the worker would take it straight back from a waiting ingest."""
    with ACC_LOCK:
        yield
    time.sleep(2e-4)


def _cache_get(key) -> ObfSetup | None:
    with _SETUPS_LOCK:
        if key in _SETUPS:
            _SETUPS.move_to_end(key)
            return _SETUPS[key]
    return None


def _cache_put(key, setup: ObfSetup) -> None:
    """Keep the last `SETUP_CACHE_SIZE` setups, so a retune and back finds the Gram again."""
    with _SETUPS_LOCK:
        _SETUPS[key] = setup
        while len(_SETUPS) > SETUP_CACHE_SIZE:
            _SETUPS.popitem(last=False)


def _prefix(ok: np.ndarray) -> int:
    """How many leading entries of `ok` are true."""
    return int(np.argmin(ok)) if not ok.all() else int(ok.size)


def _longest_run(rows: np.ndarray) -> list[int] | None:
    """[r0, r1) of the longest run of true rows, or None for none."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    if edges.size == 0:
        return None
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return [int(starts[k]), int(ends[k])]


class _Interrupted(Exception):
    """The geometry changed, OBF was switched off or the follower closed during a sweep."""


class ObfFollower:
    """Follow a `latest` `ParallaxAccumulator` with OBF. Attaches itself (`acc.obf`) and starts its
    thread; `close()` stops it. `enabled`: reconstruct OBF (the channel); `precompute`: keep the
    setup, the Gram above all, built for the current geometry also while OBF is off."""

    def __init__(self, acc, *, enabled: bool = True, precompute: bool = False, chunk: int = 2048,
                 live_every: float = 0.2, budget: float = 0.3, renorm_every: float = 10.0,
                 renorm_drift: float = 0.02, fft_sizes: bool = False, builder=None):
        if acc.accumulate != "latest":
            raise ValueError("OBF follows the frames of --accumulate latest")
        self.acc = acc
        self.device = acc.device
        self.n = int(acc.n_positions)
        self.scan_shape = tuple(int(v) for v in acc.scan_shape)
        self.chunk = int(chunk)
        self.live_every = float(live_every)
        self.budget = float(budget)                  # s of deposits between two readouts at most
        self.renorm_every = float(renorm_every)      # s between checks of the bright-field means
        self.renorm_drift = float(renorm_drift)      # their rms relative change that renews them
        self.fft_sizes = bool(fft_sizes)
        self._builder = builder or ObfSetup.build    # (geometry, device, dtype, fft_sizes)
        self._lock = threading.Lock()                # the published image and the requests
        self._wake = threading.Event()
        self._closed = False
        self._enabled = bool(enabled)
        self.precompute = bool(precompute)
        self._building = False                       # a setup is being built
        self._geometry = acc.geometry
        self._final = None                           # an Event while finalize() waits
        self.state = "building" if enabled else "off"
        self.note = ""
        # the published image, the output rows it covers (None: all), and when
        self.image_np: np.ndarray | None = None
        self.done_np: np.ndarray | None = None
        self._image_time = 0.0
        self._scale = None                           # the stand-in's scale to OBF's contrast
        self._last_publish = 0.0
        self.sweep_s = None                          # the last re-sweep (finalize), s
        self.rate_khz = None                         # positions deposited per second, lately
        self.coverage = 0.0                          # the share of the positions held covered
        # the worker's own (only it touches these)
        self._setup: ObfSetup | None = None
        self._buffer = None
        self._shadow = None                          # (positions, M) raw values deposited
        self._seen = np.zeros(self.n, dtype=bool)    # positions covered
        self._lo = self._hi = 0                      # the covered stretch, [lo, hi) in raster order
        self._cursor = 0                             # round-robin start of the updates
        self._kind = None
        self._mean = None
        self._mean_count = 0
        self._mean_time = 0.0
        self._held = 0                               # positions held at the last readout
        self._exact = False                          # the means are those of the shadow as it is
        self._changed = False                        # deposits since the last image
        self._image_s = 0.0                          # the last readout, s
        self._base = None                            # the readout's Butterworth / Nyquist filter
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        with ACC_LOCK:
            acc.obf = self
            if enabled:
                acc._obf_dirty = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self._thread = threading.Thread(target=self._run, name="obf-follower", daemon=True)
        self._thread.start()

    # ── called from other threads ──

    def set_geometry(self, geometry) -> None:
        """A retune (under ACC_LOCK): the weights are rebuilt in the background and the coverage
        restarts; the channel shows the parallax stand-in meanwhile."""
        with self._lock:
            self._geometry = geometry
        self._wake.set()

    def set_enabled(self, on: bool) -> str:
        """Switch OBF on or off (under ACC_LOCK). Off frees the shadow and the image."""
        on = bool(on)
        if on == self._enabled:
            return f"OBF already {'on' if on else 'off'}"
        with ACC_LOCK:
            self._enabled = on
            self.acc._obf_dirty = (torch.zeros(self.n, dtype=torch.bool, device=self.device)
                                   if on else None)
        self._wake.set()
        return "OBF on" if on else "OBF off"

    def set_precompute(self, on: bool) -> None:
        """Keep the setup built for the current geometry while OBF is off, or not."""
        self.precompute = bool(on)
        self._wake.set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def weights_state(self) -> str:
        """`ready` when the setup (the Gram) is that of the current geometry, `building` while it
        is built or due, `off` when nothing asks for it."""
        if self._building:
            return "building"
        if self._setup is not None and self._setup.key == geometry_key(self._geometry):
            return "ready"
        return "building" if self._enabled or self.precompute else "off"

    def image(self) -> np.ndarray | None:
        """The latest OBF image (host float32, the parallax grid; zero where not yet covered)."""
        with self._lock:
            return self.image_np if self._enabled else None

    def meta(self) -> dict:
        meta = {"obf_state": self.state, "obf_weights_state": self.weights_state(),
                "obf_coverage": round(self.coverage, 4)}
        if self.image_np is not None:
            meta["obf_age_s"] = round(time.monotonic() - self._image_time, 3)
        if self.rate_khz is not None:
            meta["obf_rate_khz"] = round(self.rate_khz, 2)
        if self.sweep_s is not None:
            meta["obf_sweep_s"] = round(self.sweep_s, 3)
        return meta

    def compose(self, standin: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
        """The channel to show: OBF on the output rows it covers, `standin` (the phase-flipped
        parallax image of the same frames) elsewhere, scaled to OBF's contrast over the rows both
        cover; and [r0, r1), the longest run of OBF rows, while it is partial (None when all OBF).
        """
        with self._lock:
            image, done = self.image_np, self.done_np
        standin = np.asarray(standin, dtype=np.float32)
        if image is None:
            if self._scale is None:
                return standin, None
            return (self._scale * (standin - standin.mean())).astype(np.float32), None
        if done is None:
            return image, None
        if done.sum() >= 4:                  # match the stand-in to OBF where both are there
            spread = float(standin[done].std())
            if spread > 0:
                self._scale = float(image[done].std()) / spread
            offset = float(standin[done].mean()) - float(image[done].mean()) / (self._scale or 1.0)
        else:
            offset = float(standin.mean())
        scaled = (self._scale or 1.0) * (standin - offset)
        return np.where(done[:, None], image, scaled).astype(np.float32), _longest_run(done)

    def finalize(self, timeout: float = 120.0) -> np.ndarray | None:
        """Wait until the image is that of every frame held, with exact bright-field means (the
        image `reconstruct_once` gives for the store), and return it; None if OBF is off or the
        wait timed out."""
        if not self._enabled or self._closed:
            return None
        done = threading.Event()
        with self._lock:
            self._final = done
        self._wake.set()
        if not done.wait(timeout):
            print(f"[obf] the final image is not ready after {timeout:.0f} s ({self.state}, "
                  f"{100 * self.coverage:.0f}% covered); saving the last one", flush=True)
        with self._lock:
            return self.image_np

    def close(self, wait: bool = True) -> None:
        """Stop the thread (`wait`: and join it; not while holding ACC_LOCK, which it may need to
        reach its next check)."""
        self._closed = True
        self._wake.set()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)

    # ── the worker ──

    def _run(self) -> None:
        while not self._closed:
            try:
                self._step()
            except _Interrupted:
                continue
            except Exception as exc:   # report, keep the pipeline running, retry after a pause
                self.state, self.note = "error", f"{type(exc).__name__}: {exc}"
                print(f"[obf] error, OBF paused: {self.note}", flush=True)
                traceback.print_exc()
                self._release(None)
                self._wake.wait(5.0)
                self._wake.clear()

    def _gpu(self):
        """Where the worker's device work runs: on its own stream (CUDA); under ACC_LOCK a
        sub-batch at a time (MPS, whose command queue takes one thread at a time); as is (CPU)."""
        if self._stream is not None:
            return torch.cuda.stream(self._stream)
        if self.device.type == "mps":
            return _turn()
        return contextlib.nullcontext()

    def _handoff(self, *tensors) -> None:
        """Tensors made on the default stream under the lock, used next on the worker's."""
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.default_stream(self.device))
            for t in tensors:
                t.record_stream(self._stream)

    def _step_size(self, setup: ObfSetup) -> int:
        """Positions per unit of device work: one deposit sub-batch on MPS, where each unit holds
        the ingest off, else a chunk."""
        return min(self.chunk, setup._batch) if self.device.type == "mps" else self.chunk

    def _check(self, setup: ObfSetup | None = None) -> None:
        if self._closed or not self._enabled:
            raise _Interrupted
        if setup is not None and geometry_key(self._geometry) != setup.key:
            raise _Interrupted

    def _step(self) -> None:
        if not self._enabled:
            if self._shadow is not None or self.image_np is not None:
                with self._gpu():
                    self._reset()
                with self._lock:
                    self.image_np = self.done_np = None
            self.state = "off"
            if self.precompute:                      # the weights: kept for the geometry
                self._ensure_setup()
            self._idle()
            return
        setup = self._ensure_setup()
        if self._buffer is None and not self._restart(setup):
            self._idle()                             # no frames held yet
            return
        final = self._final
        step = self._step_size(setup)
        t0, n_done, n = time.perf_counter(), 0, 0
        while True:
            self._check(setup)
            work = self._gather(setup)
            n = self._fold(setup, *work) if work is not None else 0
            n_done += n
            # a little at a time (following the scan): at the live cadence, not chunk by chunk,
            # which would take the lock from the ingest for every few frames
            if final is None and (n < step // 4 or time.perf_counter() - t0 > self.budget):
                break
            if not n:
                break
        if n_done:
            self._sync()
            self.rate_khz = n_done / max(time.perf_counter() - t0, 1e-9) / 1e3
        if final is not None:
            if not self._exact:
                self._resweep(setup)
            self._publish(setup, force=True)
            with self._lock:
                self._final = None
            self._release(final)
            return
        if self._renew_due():
            self._buffer = None                      # new means: the coverage restarts
            return
        self._publish(setup)
        if n < step // 4:
            self._idle()

    def _release(self, final) -> None:
        if final is not None:
            final.set()
        elif self._final is not None:
            self._final.set()

    def _idle(self) -> None:
        self._wake.wait(self.live_every)
        self._wake.clear()

    def _ensure_setup(self) -> ObfSetup:
        geometry = self._geometry
        key = geometry_key(geometry)
        if self._setup is not None and self._setup.key == key:
            return self._setup
        if self._enabled:
            self.state = "building"
            with self._lock:                         # an image of another geometry: not shown
                self.image_np = self.done_np = None
        cache_key = (key, str(self.device), self.fft_sizes)
        setup = _cache_get(cache_key)
        if setup is None:
            self._building = True
            try:
                setup = self._build(geometry)
            finally:
                self._building = False
            _cache_put(cache_key, setup)
            print(f"[obf] weights for C10 = {geometry.aberration_coefs.get('C10', 0.0):.1f} Å "
                  f"built in {setup.build_s:.1f} s ({setup.placement}, canvas "
                  f"{setup.inner_shape[0]}x{setup.inner_shape[1]})", flush=True)
        if geometry_key(self._geometry) != key:        # retuned again meanwhile
            raise _Interrupted
        self._setup, self._buffer = setup, None
        self._ensure_base(geometry)
        return setup

    def _build(self, geometry) -> ObfSetup:
        if self.device.type == "mps":
            # built on the CPU, which the ingest does not use, then moved in one short hold
            built = self._builder(geometry, device="cpu", dtype=torch.float32,
                                  fft_sizes=self.fft_sizes)
            with ACC_LOCK:
                setup = built.to(self.device)
                torch.mps.synchronize()
            return setup
        with self._gpu():
            setup = self._builder(geometry, device=self.device, fft_sizes=self.fft_sizes)
        self._sync()
        return setup

    def _ensure_base(self, geometry) -> None:
        """The readout's Butterworth / Nyquist filter, the same for every geometry of a run."""
        if self._base is not None:
            return
        cfg = self.acc.readout_config
        u = int(geometry.upsampling_factor)
        base, _ = fourier_filters(
            geometry.upsampled_scan_gpts, geometry.upsampled_scan_sampling,
            geometry.wavelength, geometry.aberration_coefs, q_lowpass=cfg.q_lowpass,
            q_highpass=cfg.q_highpass, butterworth_order=cfg.butterworth_order,
            suppress_nyquist=(u > 1) if cfg.suppress_nyquist is None else cfg.suppress_nyquist)
        if base is not None:
            with self._gpu():
                self._base = torch.as_tensor(base).float().to(self.device)

    def _reset(self) -> None:
        self._buffer = self._shadow = self._mean = None
        self._seen[:] = False
        self._mean_count, self._exact, self._changed = 0, False, False
        self.coverage = 0.0

    # ── coverage ──

    def _restart(self, setup: ObfSetup) -> bool:
        """An empty numerator, the means of the frames held, and the coverage starting at the
        scan line. False while nothing is held."""
        with ACC_LOCK:
            acc = self.acc
            store, dirty = acc._store, acc._obf_dirty
            if store is None or dirty is None:
                return False
            held = torch.nonzero(acc._visits > 0).squeeze(1)
            if not held.numel():
                return False
            max_id = int(acc._max_id)
            cursor = (max_id + 1) % self.n if max_id >= 0 else 0
            dirty.zero_()                            # from now on: the positions written since
            kind = acc._store_kind
            width, dtype = store.shape[1], store.dtype
        mean = self._store_mean(held, kind, self._step_size(setup))
        with self._gpu():
            if self._shadow is None or self._shadow.shape[1] != width or \
                    self._shadow.dtype != dtype:
                self._shadow = torch.zeros((self.n, width), dtype=dtype, device=self.device)
            self._buffer = setup.buffer()
        self._seen[:] = False
        self._lo = self._hi = cursor
        self._kind, self._mean = kind, mean
        self._mean_count, self._mean_time = int(held.numel()), time.monotonic()
        self._exact, self._changed, self.coverage = False, False, 0.0
        with self._lock:
            self.image_np = self.done_np = None
        self.state = "filling"
        return True

    def _store_mean(self, held: torch.Tensor, kind: str, step: int) -> torch.Tensor:
        """(M,) each bright-field pixel's mean over the positions `held` of the store."""
        total = None
        for pos in held.split(step):
            with ACC_LOCK:
                part = _raw_float(self.acc._store[pos], kind).sum(0)
                total = part if total is None else total + part
        with ACC_LOCK:
            return (total / max(int(held.numel()), 1)).clone()

    def _gather(self, setup: ObfSetup):
        """One hold of the lock: the positions to cover at the scan line (written since the
        coverage started, from `hi` on), the frames held behind the coverage (before `lo`,
        newest first) and covered positions written again, their store rows copied and their
        marks cleared. None when there is nothing to do."""
        step = self._step_size(setup)
        with _hold():
            acc = self.acc
            dirty, store = acc._obf_dirty, acc._store
            if dirty is None or store is None:
                return None
            marked = dirty.cpu().numpy()
            held = (acc._visits > 0).cpu().numpy()
            unseen = ~self._seen
            front = (self._hi + np.arange(step)) % self.n
            front = front[:_prefix(marked[front] & unseen[front])]
            back = (self._lo - 1 - np.arange(step)) % self.n
            k_back = _prefix(held[back] & unseen[back])
            back = back[:k_back]
            back = back[~np.isin(back, front)]      # the last gap, reached from both sides
            again = np.flatnonzero(marked & self._seen)
            if again.size:                           # round-robin, so none starves
                start = int(np.searchsorted(again, self._cursor))
                again = np.concatenate((again[start:], again[:start]))[:step]
            fresh = np.concatenate((front, back))
            if not (fresh.size or again.size):
                return None
            index = torch.as_tensor(np.concatenate((fresh, again)), device=self.device)
            rows = store[index]
            dirty[index] = False
        self._hi = int((self._hi + front.size) % self.n)
        self._lo = int((self._lo - k_back) % self.n)
        if again.size:
            self._cursor = int((again[-1] + 1) % self.n)
        return fresh, again, index, rows

    def _fold(self, setup: ObfSetup, fresh: np.ndarray, again: np.ndarray, index: torch.Tensor,
              rows: torch.Tensor) -> int:
        """Deposit newly covered frames whole, and covered ones written again as new − old."""
        self._handoff(index, rows)
        sfm, k = self.acc.subtract_frame_mean, fresh.size
        with self._gpu():
            values = normalize(_raw_float(rows, self._kind), self._mean, sfm)
            if again.size:
                old = self._shadow[index[k:]]
                values[k:] -= normalize(_raw_float(old, self._kind), self._mean, sfm)
                del old
            self._shadow[index] = rows
            setup.deposit(self._buffer, values, index)
            del values, rows
        self._seen[fresh] = True
        self._exact, self._changed = False, True
        return int(index.numel())

    def _done_rows(self) -> np.ndarray:
        """(Ny·U,) the output rows that hold only covered frames: every scan row within the
        probe's radius (the largest parallax shift) and a margin is covered."""
        ny, nx = self.scan_shape
        geometry = self._geometry
        u = int(geometry.upsampling_factor)
        shifts = np.asarray(geometry.shifts, dtype=np.float64)
        reach = int(np.ceil(np.hypot(shifts[:, 0], shifts[:, 1]).max() / u)) + _REACH_MARGIN
        missing = np.concatenate(([0], np.cumsum(~self._seen.reshape(ny, nx).all(axis=1))))
        rows = np.arange(ny)
        lo, hi = np.clip(rows - reach, 0, ny), np.clip(rows + reach + 1, 0, ny)
        near = missing[hi] - missing[lo] > 0         # a scan row within reach is not covered
        return ~near[np.arange(ny * u) // u]

    def _renew_due(self) -> bool:
        """New means: while few are held, each time they double; else when they drifted."""
        if self._mean is None:
            return False
        if self._mean_count < MIN_MEAN_POSITIONS:
            return self._held >= 2 * self._mean_count
        if time.monotonic() - self._mean_time < self.renorm_every:
            return False
        self._mean_time = time.monotonic()
        mean = self._shadow_mean()
        with self._gpu():
            ratio = mean / torch.where(self._mean > 0, self._mean, torch.ones_like(self._mean))
            # a common factor (the beam current) drops out of OBF for an even χ; the rest counts
            drift = float((ratio / ratio.mean() - 1).square().mean().sqrt())
        return drift > self.renorm_drift

    def _shadow_mean(self) -> torch.Tensor:
        """(M,) each bright-field pixel's mean over the positions covered."""
        held = np.flatnonzero(self._seen)
        total = None
        for pos in np.array_split(held, max(1, -(-held.size // self.chunk))):
            with self._gpu():
                part = _raw_float(self._shadow[torch.as_tensor(pos, device=self.device)],
                                  self._kind).sum(0)
                total = part if total is None else total + part
        with self._gpu():
            return total / max(int(held.size), 1)

    def _resweep(self, setup: ObfSetup) -> None:
        """A fresh numerator from the shadow, with the exact means of the positions covered."""
        t0 = time.perf_counter()
        mean = self._shadow_mean()
        held = np.flatnonzero(self._seen)
        with self._gpu():
            buffer = setup.buffer()
        sfm = self.acc.subtract_frame_mean
        for pos in np.array_split(held, max(1, -(-held.size // self._step_size(setup)))):
            self._check(setup)
            with self._gpu():
                index = torch.as_tensor(pos, device=self.device)
                values = normalize(_raw_float(self._shadow[index], self._kind), mean, sfm)
                setup.deposit(buffer, values, index)
                del values
        self._sync()
        self._mean, self._mean_count, self._mean_time = mean, int(held.size), time.monotonic()
        self._buffer, self._exact, self._changed = buffer, True, True
        self.sweep_s = time.perf_counter() - t0

    def _publish(self, setup: ObfSetup, force: bool = False) -> None:
        """A new image, when the numerator changed since the last one, at most every
        `live_every` s and so that the readouts take at most a fifth of the worker's time; or now
        (`force`)."""
        if self._buffer is None or not self._seen.any():
            return
        interval = max(self.live_every, 4 * self._image_s)
        if not force and (not self._changed or time.monotonic() - self._last_publish < interval):
            return
        self._check(setup)
        t0 = time.perf_counter()
        with self._gpu():
            image = setup.image(self._buffer, self._base).float().cpu().numpy()
        with ACC_LOCK:
            held = (self.acc._visits > 0).cpu().numpy()
        self._image_s = time.perf_counter() - t0
        self._held = int(held.sum())
        covered = int((held & self._seen).sum())
        self.coverage = covered / max(self._held, 1)
        complete = covered == self._held
        done = None if complete else self._done_rows()
        with self._lock:
            self.image_np, self.done_np, self._image_time = image, done, time.monotonic()
        self._last_publish, self._changed = time.monotonic(), False
        self.state = "live" if complete else "filling"
        refresh = getattr(self.acc, "on_refresh", None)
        if refresh is not None:                      # the runner publishes a snapshot
            refresh()

    def _sync(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        elif self.device.type == "mps":
            with ACC_LOCK:
                torch.mps.synchronize()
