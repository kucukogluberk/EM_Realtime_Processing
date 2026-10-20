"""Parallax / tcBF engine: build the acquisition geometry (host, NumPy) and accumulate the
tilt-corrected bright-field reconstruction on the device (PyTorch), casting integer frames to
float on the GPU.

Two pieces, a clean host-prep / device-accumulate split:

- `preprocess_geometry(...)` — pure NumPy, no GPU, off the hot path. From the acquisition
  geometry (energy, convergence angle, detector/scan sampling) and the aberration coefficients
  (defocus C10, astigmatism C12/phi12) it builds (a) `bf_flat_inds` — the flat indices of the
  bright-field-disk pixels in a *center-centered* detector frame — and (b) `shifts` — the integer
  parallax shift (in upsampled-scan pixels) each BF pixel's tilted view must be moved by to
  realign it. The shift is `∇χ / 2π` in Ångstroms (χ = the aberration phase surface), i.e. for
  pure defocus simply `tilt × C10`. Fully unit-testable.

- `ParallaxAccumulator` — the hot path. Each batch of *integer* frames is uploaded as-is (or
  arrives already on the device from a Holoscan DLPack sink), cast to float on the GPU, reduced to
  its M bright-field pixels, mean-subtracted (per-frame DC removal), and **scatter-added** into the
  reconstruction with each BF pixel toroidally shifted by its precomputed parallax shift. This
  scatter-add — one `index_add_` over B·M contributions per batch — is the heavy GPU compute.

The math is ported from Georgios Varnavides's streaming-parallax reference; the `spatial_frequencies`
kx/ky naming and the `(dy, dx) = shift` unpacking are preserved so correctness can be pinned by
matching a float64 transcription of that reference in the tests (and the engine reproduces the
reference's own output on the public apoferritin dataset, Zenodo 10.5281/zenodo.18346854, to
float32 precision). Two deliberate departures from the reference script:

- The scatter uses independent `% Ny` / `% Nx` wraps — the correct toroidal boundary (the
  reference's phase-flip kernel, not ported, has a flat-wrap bug).
- The reference's `n_rot` block — which moves every BF-disk index by one detector pixel whenever
  the rotation *rounds* to a quarter turn, i.e. for any |rotation| ≥ 45° (e.g. 77.8°), not only
  for exact quarter turns — is omitted. It is even-grid bookkeeping for quarter-turn-rotated image
  data; here the disk sits at the measured `probe_center` and is never moved. For |rotation| < 45°
  the two are identical; beyond that they differ by exactly that one-pixel disk offset.

Frames are assumed **center-centered** (BF disk at `probe_center`, default the detector center),
matching real detector frames and the fftshift correction baked into `bf_flat_inds`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["ParallaxGeometry", "preprocess_geometry", "ParallaxAccumulator"]


# ── geometry (host NumPy, ported from the streaming reference) ─────────────────

def _electron_wavelength(energy: float) -> float:
    """Relativistic de Broglie wavelength in Ångstroms for an electron of `energy` eV."""
    m, e, c, h = 9.109383e-31, 1.602177e-19, 299792458.0, 6.62607e-34
    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10


def _spatial_frequencies(gpts, sampling, rotation_angle=None):
    """fftfreq grids for the detector, with an optional scan/detector rotation applied."""
    ny, nx = gpts
    sy, sx = sampling
    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    if rotation_angle is not None:
        cos_t, sin_t = np.cos(rotation_angle), np.sin(rotation_angle)
        kxa, kya = cos_t * kxa - sin_t * kya, sin_t * kxa + cos_t * kya
    return kxa, kya


def _polar_coordinates(kx, ky):
    return np.sqrt(kx**2 + ky**2), np.arctan2(ky, kx)


def _quadratic_aberration_cartesian_gradients(alpha, phi, aberration_coefs):
    """∇χ in Cartesian (x, y) for the C10/C12 aberration surface — already ×2π (see reference)."""
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    cos2, sin2 = np.cos(2.0 * (phi - phi12)), np.sin(2.0 * (phi - phi12))
    scale = 2 * np.pi
    dchi_dk = scale * alpha * (C10 + C12 * cos2)
    dchi_dphi = -scale * alpha * (C12 * sin2)

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    dchi_dx = cos_phi * dchi_dk - sin_phi * dchi_dphi
    dchi_dy = sin_phi * dchi_dk + cos_phi * dchi_dphi
    return dchi_dx, dchi_dy


@dataclass(frozen=True)
class ParallaxGeometry:
    """Preprocessed acquisition geometry for the parallax accumulator (all host-side).

    `bf_flat_inds` and `shifts` are the only things the hot path needs; the rest is metadata
    kept for readability, the result plot, and later online aberration refinement.
    """
    bf_flat_inds: np.ndarray            # (M,) int — flat idx into a center-centered (H*W) frame
    shifts: np.ndarray                  # (M, 2) int — (dy, dx) parallax shift in upsampled px
    detector_shape: tuple               # (H, W)
    scan_shape: tuple                   # (Ny, Nx) original scan grid
    upsampling_factor: int
    upsampled_scan_gpts: tuple          # (Ny*U, Nx*U) — output reconstruction shape
    wavelength: float = 0.0             # Å
    sampling: tuple = (0.0, 0.0)        # detector real-space sampling (Å)
    upsampled_scan_sampling: tuple = (0.0, 0.0)
    aberration_coefs: dict = field(default_factory=dict)

    @property
    def n_bf(self) -> int:
        return int(self.bf_flat_inds.shape[0])


def preprocess_geometry(
    det_shape: tuple[int, int],
    scan_shape: tuple[int, int],
    scan_sampling: tuple[float, float],
    energy: float,
    semiangle_cutoff: float,
    angular_sampling: tuple[float, float] | None = None,
    reciprocal_sampling: tuple[float, float] | None = None,
    aberration_coefs: dict[str, float] | None = None,
    rotation_angle: float | None = None,
    upsampling_factor: int = 1,
    probe_center: tuple[float, float] | None = None,
) -> ParallaxGeometry:
    """Build BF-pixel indices + integer parallax shifts from the acquisition geometry.

    Args:
        det_shape: (H, W) detector pixels.
        scan_shape: (Ny, Nx) raster scan grid.
        scan_sampling: (sy, sx) scan step in Ångstroms.
        energy: beam energy in eV.
        semiangle_cutoff: probe convergence semiangle in mrad (sets the BF-disk radius).
        angular_sampling | reciprocal_sampling: detector pixel size, in mrad/px or 1/Å/px
            respectively — supply exactly one (angular is the usual microscope readout).
        aberration_coefs: {"C10", "C12", "phi12"} in Å / Å / rad. Empty → in-focus (no shifts).
        rotation_angle: scan/detector relative rotation in radians (None → none).
        upsampling_factor: integer output-grid upsampling (finer shifts); 1 → scan grid.
        probe_center: BF-disk center on the detector, (row, col) px — py4DSTEM's
            (Center_X, Center_Y) from `get_probe_size`. None → the frame center.

    Returns:
        ParallaxGeometry with `bf_flat_inds` (M,) and `shifts` (M, 2), plus metadata.
    """
    if (angular_sampling is None) == (reciprocal_sampling is None):
        raise ValueError("Specify exactly one of `angular_sampling` or `reciprocal_sampling`.")
    if aberration_coefs is None:
        aberration_coefs = {}

    wavelength = _electron_wavelength(energy)
    if reciprocal_sampling is None:
        reciprocal_sampling = (angular_sampling[0] / wavelength / 1e3,
                               angular_sampling[1] / wavelength / 1e3)

    gpts = (int(det_shape[0]), int(det_shape[1]))
    sampling = (1.0 / reciprocal_sampling[0] / gpts[0],
                1.0 / reciprocal_sampling[1] / gpts[1])
    upsampled_scan_gpts = (scan_shape[0] * upsampling_factor,
                           scan_shape[1] * upsampling_factor)
    upsampled_scan_sampling = (scan_sampling[0] / upsampling_factor,
                               scan_sampling[1] / upsampling_factor)

    kxa, kya = _spatial_frequencies(gpts, sampling, rotation_angle)
    k, phi = _polar_coordinates(kxa, kya)

    # BF disk: tilt angle α = k·λ (rad) ≤ cutoff (mrad → ×1e3). where() gives fft-order indices;
    # each maps to a signed tilt offset (px from the disk center), and the disk is placed at
    # `probe_center` in the real frame layout (None → frame center, i.e. a centered detector).
    bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
    inds_i, inds_j = np.where(bf_mask)
    off_i = ((inds_i + gpts[0] // 2) % gpts[0]) - gpts[0] // 2
    off_j = ((inds_j + gpts[1] // 2) % gpts[1]) - gpts[1] // 2
    ci = gpts[0] // 2 if probe_center is None else int(round(probe_center[0]))
    cj = gpts[1] // 2 if probe_center is None else int(round(probe_center[1]))
    det_i, det_j = ci + off_i, cj + off_j
    ok = (det_i >= 0) & (det_i < gpts[0]) & (det_j >= 0) & (det_j < gpts[1])
    if not ok.all():  # disk clips the frame edge — those tilted views fall off the detector
        inds_i, inds_j, det_i, det_j = inds_i[ok], inds_j[ok], det_i[ok], det_j[ok]
    bf_flat_inds = (det_i * gpts[1] + det_j).astype(np.int32)

    dchi_dx, dchi_dy = _quadratic_aberration_cartesian_gradients(k * wavelength, phi, aberration_coefs)
    grad_k = np.stack((dchi_dx[inds_i, inds_j], dchi_dy[inds_i, inds_j]), axis=-1)
    shifts = np.round(grad_k / (2 * np.pi) / upsampled_scan_sampling).astype(np.int32)

    return ParallaxGeometry(
        bf_flat_inds=bf_flat_inds, shifts=shifts, detector_shape=gpts,
        scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
        upsampling_factor=int(upsampling_factor), upsampled_scan_gpts=upsampled_scan_gpts,
        wavelength=wavelength, sampling=sampling,
        upsampled_scan_sampling=upsampled_scan_sampling, aberration_coefs=dict(aberration_coefs),
    )


# ── accumulator (device torch) ─────────────────────────────────────────────────

def _pick_device(device: str | None) -> torch.device:
    """Default to CUDA when available (the edge box), else CPU (so tests run anywhere)."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParallaxAccumulator:
    """Accumulate the tcBF reconstruction on the device via a batched shifted scatter-add.

    Usage:
        geom = preprocess_geometry(...)
        acc = ParallaxAccumulator(geom)
        acc.process(frames, ids)     # frames: (B, H, W) int/float; ids: (B,) scan indices
        recon = acc.result()         # (Ny*U, Nx*U) host array

    `process` accepts either host integer ndarrays or an integer/float torch tensor already on the
    device (GPU decode path, or a Holoscan DLPack sink) — the `Accumulator` contract shared by
    every algorithm.
    """

    def __init__(self, geometry: ParallaxGeometry, device: str | None = None,
                 measure: bool = False):
        self.device = _pick_device(device)
        self.geometry = geometry
        self.detector_shape = tuple(geometry.detector_shape)
        self.scan_shape = tuple(geometry.scan_shape)
        self.upsampled_scan_gpts = tuple(geometry.upsampled_scan_gpts)

        self._nx = self.scan_shape[1]                       # original scan width (ids → coords)
        self._u = int(geometry.upsampling_factor)
        self._nyu, self._nxu = self.upsampled_scan_gpts

        # Device-resident geometry: BF indices + the two shift columns (dy, dx per the reference).
        self._bf_inds = torch.as_tensor(np.asarray(geometry.bf_flat_inds), dtype=torch.long,
                                        device=self.device)
        shifts = np.asarray(geometry.shifts)
        self._sy = torch.as_tensor(shifts[:, 0], dtype=torch.long, device=self.device)
        self._sx = torch.as_tensor(shifts[:, 1], dtype=torch.long, device=self.device)

        # Flat reconstruction buffer on the device; reshaped to 2D only on readout. Flat makes the
        # shifted scatter a single index_add_.
        n = self._nyu * self._nxu
        self.out = torch.zeros(n, dtype=torch.float32, device=self.device)

        # Instrumentation (off by default): host prep, integer H2D copy, GPU cast+scatter, with
        # CUDA events so measurement doesn't perturb the run.
        self.measure = measure
        self._cuda = self.device.type == "cuda"
        self._host_prep_s = 0.0
        self._compute_s = 0.0
        self._events: list[tuple] = []

    def _to_int_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """Wrap integer frames as a torch tensor for upload, avoiding a host float cast.

        Widths take a small *integer* widening where torch lacks the unsigned dtype's kernels; the
        float conversion itself happens on the device. This host path is not the live hot path
        (the Holoscan ingest hands over device tensors), so an integer copy is fine.
        """
        a = np.ascontiguousarray(frames)
        dt = a.dtype
        if dt == np.uint32:
            # int64 widening, NOT an int32 bit-view: a bit-view turns values >= 2^31 negative,
            # and real ARINA uint32 frames carry a dtype-max (4294967295) pixel-mask sentinel.
            return torch.from_numpy(a.astype(np.int64))
        if dt == np.uint8 or dt in (np.int8, np.int16, np.int32, np.float32, np.float64):
            return torch.from_numpy(a)
        return torch.from_numpy(a.astype(np.int32))  # e.g. uint16: safe widening

    def process(self, frames, ids) -> None:
        """Upload a batch (integer), cast to float + shifted scatter-add on the GPU, by scan id.

        `ids` may be a host array-like or a torch tensor (any device) — the Holoscan sink
        passes the ingest's device tensor straight through to avoid a per-batch D2H sync."""
        if tuple(frames.shape[1:]) != self.detector_shape:
            raise ValueError(
                f"frame detector shape {tuple(frames.shape[1:])} != configured "
                f"{self.detector_shape} — the detector doesn't match the geometry."
            )

        # Already a device tensor (GPU decode / Holoscan DLPack): no host prep, no H2D here.
        if isinstance(frames, torch.Tensor):
            f = frames if frames.device == self.device else frames.to(self.device)
            if not self.measure:
                self._accumulate_into(f, ids)
                return
            if self._cuda:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                self._accumulate_into(f, ids)
                e1.record()
                self._events.append((e0, e0, e1))
            else:
                t1 = time.perf_counter()
                self._accumulate_into(f, ids)
                self._compute_s += time.perf_counter() - t1
            return

        if not self.measure:
            f_int = self._to_int_tensor(frames).to(self.device, non_blocking=True)
            self._accumulate_into(f_int, ids)
            return

        # Measured path.
        t0 = time.perf_counter()
        t_int = self._to_int_tensor(frames)
        self._host_prep_s += time.perf_counter() - t0

        if self._cuda:
            e_copy0 = torch.cuda.Event(enable_timing=True)
            e_copy1 = torch.cuda.Event(enable_timing=True)
            e_comp1 = torch.cuda.Event(enable_timing=True)
            e_copy0.record()
            g = t_int.to(self.device, non_blocking=True)
            e_copy1.record()
            self._accumulate_into(g, ids)
            e_comp1.record()
            self._events.append((e_copy0, e_copy1, e_comp1))
        else:
            t1 = time.perf_counter()
            self._accumulate_into(t_int, ids)
            self._compute_s += time.perf_counter() - t1

    def _accumulate_into(self, f: torch.Tensor, ids) -> None:
        # Gather the M bright-field pixels of every frame, remove the per-frame DC (mean over BF
        # pixels), then scatter-add each contribution to its parallax-shifted, toroidally-wrapped
        # output position. Independent % Ny / % Nx wraps = the correct scan-grid torus.
        #
        # Gather BEFORE casting to float: only M of the H·W pixels survive, so casting the whole
        # batch first would allocate and write B·H·W floats per batch for nothing.
        b = f.shape[0]
        flat_f = f.reshape(b, -1)
        if flat_f.dtype == torch.uint16:
            # uint16 is a second-class torch dtype with no CUDA gather kernel: reinterpret the
            # bits as int16 (zero-copy view), gather the M columns, then recover the exact
            # unsigned values on the small (B, M) slice. float32 holds ≤ 2^24 exactly, so
            # 0..65535 round-trips exactly.
            g16 = flat_f.view(torch.int16)[:, self._bf_inds].float()
            i_bf = torch.where(g16 < 0, g16 + 65536.0, g16)    # (B, M)
        elif flat_f.dtype == torch.uint32:
            # Same trick for uint32 (what the ingest emits for the DCU's long-dwell modes, and
            # equally second-class in torch): int32 bit-view, gather, undo the wrap on the small
            # (B, M) slice. Values >= 2^24 round in float32 exactly as a direct cast would.
            g32 = flat_f.view(torch.int32)[:, self._bf_inds].float()
            i_bf = torch.where(g32 < 0, g32 + 4294967296.0, g32)   # (B, M)
        else:
            i_bf = flat_f[:, self._bf_inds].float()            # (B, M)
        i_bf = i_bf - i_bf.mean(dim=1, keepdim=True)

        if isinstance(ids, torch.Tensor):
            # Device ids from the Holoscan sink — keep them on-device (no numpy round trip).
            idx = ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        else:
            idx = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.device)
        yt = (idx // self._nx) * self._u                       # (B,)
        xt = (idx % self._nx) * self._u
        oy = (yt.view(-1, 1) + self._sy.view(1, -1)) % self._nyu   # (B, M)
        ox = (xt.view(-1, 1) + self._sx.view(1, -1)) % self._nxu
        flat = (oy * self._nxu + ox).reshape(-1)               # (B*M,)
        self.out.index_add_(0, flat, i_bf.reshape(-1))

    def result(self) -> np.ndarray:
        """The parallax / tcBF reconstruction as a (Ny*U, Nx*U) host array."""
        return self.out.reshape(self.upsampled_scan_gpts).cpu().numpy()

    def perf_stats(self) -> dict | None:
        """Per-stage timing for the run, or None if measurement was off."""
        if not self.measure:
            return None
        out: dict = {"host_prep_s": self._host_prep_s}
        if self._cuda and self._events:
            torch.cuda.synchronize()
            out["gpu_copy_s"] = sum(a.elapsed_time(b) for a, b, _ in self._events) / 1000.0
            out["gpu_compute_s"] = sum(b.elapsed_time(c) for _, b, c in self._events) / 1000.0
            out["n_batches"] = len(self._events)
        else:
            out["compute_s"] = self._compute_s
        return out
