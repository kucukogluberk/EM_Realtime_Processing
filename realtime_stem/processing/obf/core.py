"""Optimum bright field (OBF) direct ptychography, frame by frame, on the parallax geometry.

A port of the correlation route of quantem's `DirectPtychographyMontage`
(`reconstruct(deconvolution_kernel="obf", convolution_mode="correlation")`) for the analytic
probe, one slice and a pure-phase object. quantem stays the offline oracle; nothing here imports it.

OBF's estimate at spatial frequency q is −i·b(q) / √(Gram(q)·W), W the summed bright-field
intensity. The numerator b(q) = Σ_k c_k*(q) G_k(q) is a double sum over scan positions and
bright-field pixels. The FFT route transforms each bright-field image over the scan and contracts
over k afterwards. Here the contraction comes first, frame by frame: c_k*(q) is a product of the
probe at k and at k ∓ q, so Σ_k c_k*(q) V_R(k) is a cross-correlation over the detector, which a
few padded detector FFTs give at every q at once. The work per frame therefore does not grow with
the scan, which is what makes OBF streamable.

- The frames land on a canvas aligned with the detector, a whole number m of probe fields of view
  (1/Δk) per axis, so that its lattice is the detector's refined m times. A frame reaches it either
  on the lattice (m_r·m_c ≤ 16: every sub-lattice correlated, the frame entering through its exact
  plane wave) or as a patch (otherwise, the electron case): synthesized on a window of two probe
  fields of view about its position, shifted by its sub-pixel part in Fourier space and added with
  `index_add_`. One canvas FFT at readout gives the numerator. The window misses the aperture's
  diffraction tails, about 1% of the numerator.
- A patch holds only the band the output grid can show (its Nyquist box turned into the detector
  frame, plus `BAND_MARGIN` pixels per field of view), not OBF's whole band, unless
  `margin=None` (quantem's `patch_margin`, the same rule). The patches, the detector-side
  correlations, the canvas and the Gram shrink with it. On the binned Gd2O3 scan at x2 that is
  3.8x faster per frame and 0.3% from the whole band. Each patch's spectrum is then cut at its
  edge, which is all the difference.
- The Gram Σ_k |c_k(q)|² does not depend on the data. Its direct terms are FFT autocorrelations;
  its cross term has no FFT form and is a per-pixel pass inside the probe's support radius. It is
  built once per geometry (`ObfSetup.build`), off the ingest thread.
- The readout solves per q, transforms back and reads the detector-aligned image onto the parallax
  output grid, (Ny·U, Nx·U) at the upsampled scan step, through a refine-2 cubic B-spline.

Conventions: the probe is built on parallax's k-grid (the rotation, the column flip and the disk
centre of `ParallaxGeometry`, χ of `readout.quadratic_aberration_surface`, the abTEM sign: C10 < 0
is underfocus), so OBF and parallax agree in orientation by construction. In quantem's terms the
frames are the column-flipped ("canonical") ones, its origin the disk centre after the flip, its
`rotation_angle` the rotation in degrees and its C10 the `--defocus` value.

Each bright-field pixel is normalized as quantem does, V = I/m(k) − 1 with m(k) its mean over the
positions held (`normalize`). The departures from quantem are deliberate and small:
- the detector-aligned canvas is always the rotated one, also for a rotation of zero, so the image
  is always read through the spline (quantem skips it when the requested canvas already spans
  whole fields of view, unrotated);
- the probe is evaluated in float64 on the host (quantem: float32, stored as complex64);
- the probe is evaluated at each bright-field pixel's true k, from the unrounded disk centre,
  where quantem resamples the frames onto the centre instead (`subpixel_center`);
- the per-frequency solve runs in the working precision on the device (quantem: complex128 on the
  CPU); with one column it is exact either way;
- `fft_sizes=True` rounds the canvas's pixels per field of view up to a size the FFTs handle
  quickly (a finer canvas, the same band).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from realtime_stem.processing.parallax.readout import quadratic_aberration_surface

__all__ = ["ObfSetup", "bf_offsets", "bright_field_mean", "normalize", "reconstruct_once"]

#: canvas sub-lattices (fields of view along rows times along columns) up to which a frame is
#: placed on the whole canvas in Fourier space, exactly; beyond it, as a patch
_LATTICE_SUBLATTICES = 16
#: probe fields of view per axis spanned by a patch, centred on its position
_PATCH_FIELDS = 2
#: where the Gram falls below this fraction of its peak a patch's numerator is only its error
_PATCH_FLOOR = 1e-6
#: (bright-field pixel, frequency) pairs per batch in the Gram's per-pixel pass
_GRAM_POINTS_PER_BATCH = 4_194_304
#: complex values per contraction sub-batch (quantem's `_DEFAULT_POINTS_PER_BATCH`)
_POINTS_PER_BATCH = 4_194_304
#: an overlap below this fraction of the peak, or a power below its square, is round-off
OVERLAP_FLOOR = 1e-6
#: pixels per probe field of view a patch holds beyond the output grid's Nyquist box (turned into
#: the detector frame), rounded up to an FFT size. On the binned Gd2O3 scan it is 3.8x faster per
#: frame than the whole band at x2 (2.4x at x4) and 0.3% from it (0.2% at x4); None: the band.
BAND_MARGIN = 4


# ── small helpers ──────────────────────────────────────────────────────────────

def signed_index(n: int) -> np.ndarray:
    """The signed lattice index of each corner-centred position, `fftfreq(n) * n`."""
    return np.round(np.fft.fftfreq(n, 1 / n)).astype(np.int64)


def smooth_fft_size(n: int) -> int:
    """The smallest 2^a 3^b 5^c at least `n`."""
    best = None
    power2 = 1
    while power2 < 2 * n:
        power3 = power2
        while power3 < 2 * n:
            size = power3
            while size < n:
                size *= 5
            best = size if best is None else min(best, size)
            power3 *= 3
        power2 *= 2
    return int(best)


def _reflect(x: torch.Tensor) -> torch.Tensor:
    """`x(-n)` on a periodic grid, over the last two axes."""
    return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


def _dev(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A host array on `device` as `dtype`, converted on the host first (MPS takes no float64)."""
    return torch.as_tensor(array).to(dtype).to(device)


def _working_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    """float32 unless asked otherwise; MPS has no float64."""
    if device.type == "mps" or dtype is None:
        return torch.float32
    return dtype


def bf_offsets(geometry) -> np.ndarray:
    """(M, 2) signed detector offsets (row, col) of the bright-field pixels from k = 0, in the
    canonical (column-unflipped) frame, in the order of `geometry.bf_flat_inds`: the inverse of
    the placement in `preprocess_geometry`."""
    h, w = (int(n) for n in geometry.detector_shape)
    det_i, det_j = np.divmod(np.asarray(geometry.bf_flat_inds, dtype=np.int64), w)
    center = geometry.probe_center
    ci = h // 2 if center is None else int(round(center[0]))
    if center is not None:
        cj = int(round(center[1]))
    else:
        cj = w - 1 - w // 2 if geometry.flip_cols else w // 2
    off_j = cj - det_j if geometry.flip_cols else det_j - cj
    return np.stack((det_i - ci, off_j), axis=-1)


def center_fraction(geometry) -> np.ndarray:
    """(2,) the true k of offset 0, in detector pixels of the canonical frame: `preprocess_geometry`
    rounds the disk centre to a pixel, so each bright-field pixel sits this far from where the
    rounding puts it."""
    center = geometry.probe_center
    if center is None:
        return np.zeros(2)
    frac = np.asarray(center, dtype=np.float64) - np.round(center)
    # a raw column right of the centre is a canonical column left of it
    return np.array([-frac[0], frac[1] if geometry.flip_cols else -frac[1]])


def _raw_float(raw: torch.Tensor, kind: str) -> torch.Tensor:
    """Stored (gathered) bright-field values → float32, undoing the unsigned bit-views of
    `ParallaxAccumulator._gather`."""
    g = raw.float()
    if kind == "u16":
        g = torch.where(g < 0, g + 65536.0, g)
    elif kind == "u32":
        g = torch.where(g < 0, g + 4294967296.0, g)
    return g


def bright_field_mean(store: torch.Tensor, kind: str, held: torch.Tensor,
                      chunk: int = 4096) -> torch.Tensor:
    """(M,) mean of each bright-field pixel over the positions `held` of a `latest` store."""
    total = torch.zeros(store.shape[1], dtype=torch.float64 if store.device.type != "mps"
                        else torch.float32, device=store.device)
    for pos in held.split(chunk):
        total += _raw_float(store[pos], kind).sum(0).to(total.dtype)
    return (total / max(int(held.numel()), 1)).float()


def normalize(values: torch.Tensor, mean: torch.Tensor,
              subtract_frame_mean: bool = False) -> torch.Tensor:
    """quantem's V = I/m − 1 per bright-field pixel (zero where m is), then, with
    `subtract_frame_mean`, each frame's mean over its bright-field pixels removed."""
    safe = torch.where(mean > 0, mean, torch.ones_like(mean))
    v = torch.where(mean > 0, values / safe - 1.0, torch.zeros_like(values))
    if subtract_frame_mean:
        v = v - v.mean(dim=1, keepdim=True)
    return v


# ── the probe and the canvas layout (host, NumPy) ─────────────────────────────

def _probe(shape, step, wavelength, semiangle, angular_sampling, coefs, rotation,
           shift=(0.0, 0.0)) -> np.ndarray:
    """ψ(k) = A(k)·exp(−iχ(k)) on a detector-aligned lattice of `shape` points `step` 1/Å apart,
    corner-centred, read at the scan-frame frequency R·k of each: quantem's analytic
    `FourierProbe.at` with its soft-edged aperture (a pixel wide at the detector's own
    `angular_sampling`, mrad/px). `shift` (1/Å, detector frame) is the true k of lattice point 0,
    for a disk centre between pixels."""
    kx = signed_index(shape[0]) * step[0] + shift[0]
    ky = signed_index(shape[1]) * step[1] + shift[1]
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")
    c, s = math.cos(rotation), math.sin(rotation)
    kxa, kya = c * kxa - s * kya, s * kxa + c * kya
    alpha = np.hypot(kxa, kya) * wavelength
    phi = np.arctan2(kya, kxa)
    width = np.sqrt((np.cos(phi) * angular_sampling[0] * 1e-3) ** 2
                    + (np.sin(phi) * angular_sampling[1] * 1e-3) ** 2)
    aperture = np.clip((semiangle * 1e-3 - alpha) / width + 0.5, 0.0, 1.0)
    chi = quadratic_aberration_surface(alpha, phi, wavelength, coefs)
    return aperture * np.exp(-1j * chi)


def _support(probe: np.ndarray) -> np.ndarray:
    """Where |ψ|² exceeds round-off, OVERLAP_FLOOR² of its peak."""
    power = np.abs(probe) ** 2
    return power > OVERLAP_FLOOR**2 * power.max()


def _layout(canvas_shape, sampling, fov, reach, rotation, margin=None):
    """Where the frames land (quantem's `_correlation_layout`, its rotated branch): `multiples`
    probe fields of view per axis span a canvas of `per_fov` pixels to a field of view, covering
    the requested canvas turned into the detector frame, with room per side for half a field of
    view (lattice) or half a patch, so nothing wraps onto it. `_finish_layout` adds the rest.

    A patch holds OBF's whole band if `margin` is None. Otherwise it holds only the requested
    canvas's Nyquist box turned into the detector frame and `margin` pixels per field of view
    beyond it, so the patches and the detector-side correlations shrink, but never fewer pixels
    than the requested canvas has (quantem's `patch_margin`)."""
    sampling = np.asarray(sampling, dtype=np.float64)
    band = 4 * reach + 1                      # pixels per field of view holding the whole band
    native = fov / sampling
    rot = np.array([[math.cos(rotation), -math.sin(rotation)],
                    [math.sin(rotation), math.cos(rotation)]])
    extent = np.asarray(canvas_shape, dtype=np.float64) * sampling
    corners = np.array([[0, 0], [extent[0], 0], [0, extent[1]], extent]) @ rot
    span = corners.max(0) - corners.min(0)
    multiples = np.ceil(span / fov + 1 - 1e-9).astype(int)
    lattice = int(np.prod(multiples)) <= _LATTICE_SUBLATTICES
    if not lattice:
        multiples = np.ceil(span / fov + _PATCH_FIELDS - 1e-9).astype(int)
    # pixels no coarser than the requested ones, fine enough for the requested canvas's Nyquist
    # box turned into this frame, never beyond the band (except that a patch needs the band)
    turned = fov * (np.abs(rot[0, :]) / sampling[0] + np.abs(rot[1, :]) / sampling[1])
    per_fov = np.maximum(np.ceil(native - 1e-6), np.minimum(np.ceil(turned - 1e-6), band))
    if not lattice:
        per_fov = np.maximum(per_fov, band if margin is None else np.minimum(
            [smooth_fft_size(int(v)) for v in per_fov + int(margin)], band))
    return dict(placement="lattice" if lattice else "patch", multiples=multiples,
                per_fov=per_fov.astype(int), rotation=rot, corners=corners, span=span)


def _finish_layout(layout, fov, positions_A):
    """The canvas `shape` of pixels `step` Å wide, its corner `origin` in Å in the detector
    frame, and the scan `positions` (Å) in its pixels, once `per_fov` is settled."""
    multiples, per_fov, span = layout["multiples"], layout["per_fov"], layout["span"]
    step = fov / per_fov
    # centred to a whole pixel, so an unrotated canvas with the same pixels is read at whole pixels
    origin = layout["corners"].min(0) - np.round((multiples * fov - span) / 2 / step) * step
    layout.update(shape=tuple(int(v) for v in multiples * per_fov), step=step, origin=origin,
                  positions=(positions_A @ layout["rotation"] - origin) / step)
    return layout


# ── the correlation operands ───────────────────────────────────────────────────

@dataclass
class _Operands:
    """A correlation onto a grid whose lattice is `sub` times the detector's: the detector
    grid `padded` so no lag up to `lags` wraps, the probe shifted by s/sub of a pixel per
    sub-lattice s (`minus`: ψ(k − sΔk/sub), `plus`: ψ(k + sΔk/sub), (S, *padded)), and the map
    of per-sub-lattice correlations onto the grid (zero beyond `lags`)."""
    padded: tuple
    sub: tuple
    minus: torch.Tensor
    plus: torch.Tensor
    row_lookup: torch.Tensor
    col_lookup: torch.Tensor
    kept: torch.Tensor | None

    @property
    def psi(self) -> torch.Tensor:
        return self.minus[0]

    def to_grid(self, stack: torch.Tensor) -> torch.Tensor:
        """(S, *lead, *padded) per-sub-lattice correlations → (*lead, *grid)."""
        s0, s1 = self.sub
        p0, p1 = self.padded
        lead = stack.shape[1:-2]
        stack = stack.reshape(s0, s1, *lead, p0, p1).movedim((0, 1), (-4, -2))
        stack = stack.reshape(*lead, s0 * p0, s1 * p1)
        out = stack.index_select(-2, self.row_lookup).index_select(-1, self.col_lookup)
        return out if self.kept is None else out * self.kept


def _operands(refined: np.ndarray, gpts, sub, grid_shape, lags, device, cdtype) -> _Operands:
    """quantem's `_lattice_operands` + `_sublattice_probe` for one coherent mode."""
    padded = tuple(smooth_fft_size(int(gpts[a]) + int(lags[a]) + 1) for a in (0, 1))
    rows = signed_index(int(gpts[0])) % padded[0]
    cols = signed_index(int(gpts[1])) % padded[1]
    out = {}
    for sign in (-1, 1):
        stack = np.zeros((int(sub[0]) * int(sub[1]), *padded), dtype=np.complex128)
        for n, (a, b) in enumerate((a, b) for a in range(int(sub[0])) for b in range(int(sub[1]))):
            picks = []
            for axis, s in ((0, a), (1, b)):
                size, factor = int(gpts[axis]), int(sub[axis])
                index = factor * signed_index(size) + sign * s
                valid = (index >= -(size * factor // 2)) & (index < (size * factor + 1) // 2)
                picks.append((index % (size * factor), valid))
            (ri, rv), (ci, cv) = picks
            stack[n][rows[:, None], cols[None, :]] = refined[ri[:, None], ci[None, :]] * (
                rv[:, None] & cv[None, :])
        out[sign] = _dev(stack, device, cdtype)

    def lookup(n, factor, size, protected):
        index = signed_index(n)
        lag = index // factor
        flat = (index % factor) * size + lag % size
        return (torch.as_tensor(flat, device=device),
                torch.as_tensor(np.abs(lag) <= protected, device=device))

    row_lookup, row_kept = lookup(int(grid_shape[0]), int(sub[0]), padded[0], int(lags[0]))
    col_lookup, col_kept = lookup(int(grid_shape[1]), int(sub[1]), padded[1], int(lags[1]))
    kept = row_kept[:, None] & col_kept[None, :]
    return _Operands(padded=padded, sub=(int(sub[0]), int(sub[1])), minus=out[-1], plus=out[1],
                     row_lookup=row_lookup, col_lookup=col_lookup,
                     kept=None if bool(kept.all()) else kept)


# ── the Gram ───────────────────────────────────────────────────────────────────

def _gram(refined: np.ndarray, operands: _Operands, offsets: np.ndarray, gpts, multiples,
          grid_shape, dk, device, rdtype, cdtype) -> torch.Tensor:
    """Σ_k |c_k(q)|² on the canvas lattice (quantem's `_correlation_gram`, one mode).

    |c|² = |t−|² + |t+|² − 2 Re(t− t+*). Summed over k the first two are autocorrelations of
    |ψ|², one FFT pair per sub-lattice. The cross term Σ_k ψ(k − q) ψ(k + q) ψ*(k)² has no FFT
    form; it is evaluated per bright-field pixel where it can be nonzero, |q| within the probe's
    support radius, and |c_k(−q)| = |c_k(q)| halves that. The Gram must be exact on the grid it
    is used on: it falls steeply at the band's edge, where OBF divides by its square root."""
    psi = operands.psi
    p0 = torch.fft.fft2(psi.abs().square().to(cdtype))
    minus = torch.fft.fft2(operands.minus.abs().square().to(cdtype)).conj() * p0
    plus = torch.fft.fft2(operands.plus.abs().square().to(cdtype)).conj() * p0
    gram = (operands.to_grid(torch.fft.ifft2(minus))
            + operands.to_grid(_reflect(torch.fft.ifft2(plus)))).real.to(rdtype)
    del minus, plus

    # the cross term, per bright-field pixel, where it can live
    padded = operands.padded
    support = psi.abs().square() > OVERLAP_FLOOR**2 * psi.abs().square().max()
    sr, sc = signed_index(int(gpts[0])), signed_index(int(gpts[1]))
    on_det = support[torch.as_tensor(sr % padded[0], device=device)[:, None],
                     torch.as_tensor(sc % padded[1], device=device)[None, :]].cpu().numpy()
    radius = float(np.hypot(sr[:, None] * dk[0], sc[None, :] * dk[1])[on_det].max())
    g0, g1 = (int(n) for n in grid_shape)
    rows = np.broadcast_to(signed_index(g0)[:, None], (g0, g1))
    cols = np.broadcast_to(signed_index(g1)[None, :], (g0, g1))
    m0, m1 = (int(v) for v in multiples)
    inside = np.hypot(rows * dk[0] / m0, cols * dk[1] / m1) <= radius * (1 + 1e-6)
    upper = (rows >= 0) | (rows == -(g0 // 2)) | (cols == -(g1 // 2))
    computed = inside & upper
    if computed.any():
        q_row = torch.as_tensor(rows[computed], device=device)
        q_col = torch.as_tensor(cols[computed], device=device)
        k_row = torch.as_tensor(offsets[:, 0] * m0, device=device)
        k_col = torch.as_tensor(offsets[:, 1] * m1, device=device)
        # quantem's LatticeOverlap: the refined probe fftshifted into a zero-padded flat array
        # wide enough that every k ∓ q lands inside it, so a batch is one add and one gather
        n0, n1 = refined.shape
        reach0, reach1 = int(np.abs(rows[computed]).max()) + 1, int(np.abs(cols[computed]).max()) + 1
        width = n1 + 2 * reach1
        flat = np.zeros((n0 + 2 * reach0, width), dtype=np.complex128)
        flat[reach0:reach0 + n0, reach1:reach1 + n1] = np.fft.fftshift(refined)
        flat = _dev(flat.reshape(-1), device, cdtype)
        k_base = (k_row + reach0 + n0 // 2) * width + (k_col + reach1 + n1 // 2)
        q_offset = q_row * width + q_col
        exact = torch.zeros(q_offset.numel(), dtype=rdtype, device=device)
        batch = max(1, _GRAM_POINTS_PER_BATCH // q_offset.numel())
        for lo in range(0, k_base.numel(), batch):
            base = k_base[lo:lo + batch, None]
            psi_k = flat[base]
            t_minus = flat[base - q_offset[None]] * psi_k.conj()
            t_plus = flat[base + q_offset[None]].conj() * psi_k
            exact += (t_minus - t_plus).abs().square().sum(0).to(rdtype)
        gram[torch.as_tensor(computed, device=device)] = exact
        mirror = inside & ~upper
        src = ((-rows[mirror]) % g0, (-cols[mirror]) % g1)
        gram[torch.as_tensor(mirror, device=device)] = gram[
            torch.as_tensor(src[0], device=device), torch.as_tensor(src[1], device=device)]
    return gram


# ── reading the detector-aligned image onto the parallax grid ──────────────────

class _CanvasReader:
    """quantem's `_read_onto_canvas`, with everything that does not depend on the image built
    once: the image's spectrum is cut to the requested canvas's Nyquist box turned into the
    detector frame, refined `refine` times by zero-padding, divided by the transform of the
    sampled cubic B-spline, and the spline read at each requested pixel."""

    def __init__(self, inner_shape, step, origin, rotation, canvas_shape, sampling, device,
                 rdtype, cdtype, refine: int = 2):
        self.device, self.rdtype, self.cdtype = device, rdtype, cdtype
        n_rows, n_cols = inner_shape
        q_rows = np.fft.fftfreq(n_rows, step[0])[:, None]
        q_cols = np.fft.fftfreq(n_cols, step[1])[None, :]
        along_rows = q_rows * rotation[0, 0] + q_cols * rotation[0, 1]
        along_cols = q_rows * rotation[1, 0] + q_cols * rotation[1, 1]
        keep = ((np.abs(along_rows) <= 0.5 / sampling[0] * (1 + 1e-9))
                & (np.abs(along_cols) <= 0.5 / sampling[1] * (1 + 1e-9)))
        # an even size's Nyquist bin has no partner on the finer grid
        if n_rows % 2 == 0:
            keep[n_rows // 2] = False
        if n_cols % 2 == 0:
            keep[:, n_cols // 2] = False
        self.keep = torch.as_tensor(keep, device=device)
        self.fine = (refine * n_rows, refine * n_cols)
        self.rows = torch.as_tensor(signed_index(n_rows) % self.fine[0], device=device)
        self.cols = torch.as_tensor(signed_index(n_cols) % self.fine[1], device=device)
        self.spline = [_dev((2 + np.cos(2 * np.pi * np.fft.fftfreq(n))) / 3, device, rdtype)
                       for n in self.fine]
        self.scale = float(refine * refine)
        i = np.arange(canvas_shape[0], dtype=np.float64)[:, None] * sampling[0]
        j = np.arange(canvas_shape[1], dtype=np.float64)[None, :] * sampling[1]
        u = (i * rotation[0, 0] + j * rotation[1, 0] - origin[0]) / (step[0] / refine)
        v = (i * rotation[0, 1] + j * rotation[1, 1] - origin[1]) / (step[1] / refine)
        self.taps = []
        for coordinate, n in ((u, self.fine[0]), (v, self.fine[1])):
            base = np.floor(coordinate)
            t = _dev(coordinate - base, device, rdtype)
            weights = ((1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6,
                       (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6)
            index = [torch.as_tensor((base.astype(np.int64) + o) % n, device=device)
                     for o in (-1, 0, 1, 2)]
            self.taps.append((weights, index))

    def to(self, device: torch.device) -> "_CanvasReader":
        new = object.__new__(_CanvasReader)
        new.__dict__.update(self.__dict__)
        new.device = device
        new.keep, new.rows, new.cols = (t.to(device) for t in (self.keep, self.rows, self.cols))
        new.spline = [t.to(device) for t in self.spline]
        new.taps = [(tuple(w.to(device) for w in weights), [i.to(device) for i in index])
                    for weights, index in self.taps]
        return new

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(image.to(self.cdtype)) * self.keep
        refined = torch.zeros(self.fine, device=self.device, dtype=self.cdtype)
        refined[self.rows[:, None], self.cols[None, :]] = spectrum
        refined /= self.spline[0][:, None] * self.spline[1][None, :]
        coefficients = torch.fft.ifft2(refined).real * self.scale
        (rw, ri), (cw, ci) = self.taps
        out = torch.zeros(rw[0].shape, device=self.device, dtype=self.rdtype)
        for a in range(4):
            for b in range(4):
                out += rw[a] * cw[b] * coefficients[ri[a], ci[b]]
        return out


# ── the setup: everything that does not depend on the data ─────────────────────

def geometry_key(geometry) -> tuple:
    """What an `ObfSetup` depends on: the probe, the bright-field pixels, the rotation and the
    output grid. Equal keys share a setup (and its Gram)."""
    coefs = geometry.aberration_coefs
    return (tuple(geometry.detector_shape), tuple(geometry.scan_shape),
            tuple(geometry.upsampled_scan_gpts), tuple(round(float(v), 9)
                                                       for v in geometry.upsampled_scan_sampling),
            round(float(geometry.wavelength), 12), round(float(geometry.semiangle_cutoff), 9),
            tuple(round(float(v), 12) for v in geometry.sampling),
            round(float(coefs.get("C10", 0.0)), 6), round(float(coefs.get("C12", 0.0)), 6),
            round(float(coefs.get("phi12", 0.0)), 9),
            round(float(geometry.rotation_angle or 0.0), 12),
            np.asarray(geometry.bf_flat_inds).tobytes(), bool(geometry.flip_cols),
            None if geometry.probe_center is None else tuple(geometry.probe_center))


@dataclass
class ObfSetup:
    """The probe, the canvas layout, the correlation operands and the Gram for one geometry.

    `buffer()` makes an empty numerator; `deposit(buffer, values, pos)` adds frames (normalized
    bright-field values (B, M), scan positions (B,)); `image(buffer)` reads the OBF image onto
    the parallax output grid. Deposits are linear in the values and independent of order, so a
    numerator can be split across buffers and summed (`latest`: deposit new − old)."""

    key: tuple
    device: torch.device
    rdtype: torch.dtype
    cdtype: torch.dtype
    placement: str
    multiples: tuple
    per_fov: tuple
    inner_shape: tuple
    grid_shape: tuple
    canvas_shape: tuple
    n_bf: int
    bf_weights: float
    gram: torch.Tensor
    build_s: float = 0.0
    _ops: _Operands | None = field(default=None, repr=False)
    _bf_flat: torch.Tensor | None = field(default=None, repr=False)
    _conj_a_hat: torch.Tensor | None = field(default=None, repr=False)
    _b_hat: torch.Tensor | None = field(default=None, repr=False)
    _q: tuple | None = field(default=None, repr=False)
    _place: tuple | None = field(default=None, repr=False)
    _offsets: tuple | None = field(default=None, repr=False)
    _operator: torch.Tensor | None = field(default=None, repr=False)
    _numerator_mask: torch.Tensor | None = field(default=None, repr=False)
    _reader: _CanvasReader | None = field(default=None, repr=False)
    _scale: float = 1.0
    _batch: int = 1

    @classmethod
    def build(cls, geometry, device=None, dtype: torch.dtype | None = None,
              fft_sizes: bool = False, subpixel_center: bool = True,
              margin: int | None = BAND_MARGIN) -> "ObfSetup":
        """Everything OBF needs for `geometry` (a `ParallaxGeometry`), on `device`, in `dtype`
        (float32 by default; float64 for checks off MPS). Takes seconds (the Gram): build it off
        the ingest thread. `subpixel_center` evaluates the probe at each bright-field pixel's true
        k, from the unrounded disk centre (`center_fraction`); off, the probe sits on the rounded
        centre as parallax's shifts do. On the Gd2O3 recordings (0.14 and 0.41 px off a pixel) it
        takes the interior correlation with quantem's own OBF from 0.988 to 0.999. `margin`: a
        patch holds only the output grid's band and this many pixels per field of view beyond it
        (`BAND_MARGIN`); None, OBF's whole band (quantem's `patch_margin=None`)."""
        import time
        t0 = time.perf_counter()
        device = torch.device(device) if device is not None else torch.device("cpu")
        rdtype = _working_dtype(device, dtype)
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        offsets = bf_offsets(geometry)
        reach_bf = np.abs(offsets).max(0)
        gpts = tuple(int(2 * a + 3) for a in reach_bf)          # the bright-field crop, ±1 px
        dk = np.array([1.0 / (geometry.detector_shape[a] * geometry.sampling[a]) for a in (0, 1)])
        fov = 1.0 / dk
        wavelength = float(geometry.wavelength)
        angular = dk * wavelength * 1e3                         # mrad/px
        rotation = float(geometry.rotation_angle or 0.0)
        coefs = dict(geometry.aberration_coefs)
        shift = center_fraction(geometry) * dk if subpixel_center else np.zeros(2)

        def refined(m):
            m = np.asarray(m, dtype=int)
            return _probe(tuple(int(g * f) for g, f in zip(gpts, m)), dk / m, wavelength,
                          float(geometry.semiangle_cutoff), angular, coefs, rotation, shift)

        fields = np.full(2, _PATCH_FIELDS)
        patch_probe = refined(fields)
        support = _support(patch_probe)
        reach = []
        for axis in (0, 1):
            index = signed_index(gpts[axis] * int(fields[axis]))
            line = support.any(axis=1 - axis)
            reach.append(max(math.ceil(float(np.abs(index[line]).max()) / fields[axis]),
                             int(reach_bf[axis])))
        reach = np.asarray(reach)

        sampling = np.asarray(geometry.upsampled_scan_sampling, dtype=np.float64)
        scan_sampling = sampling * int(geometry.upsampling_factor)
        ny, nx = (int(n) for n in geometry.scan_shape)
        iy, ix = np.divmod(np.arange(ny * nx), nx)
        positions_A = np.stack((iy * scan_sampling[0], ix * scan_sampling[1]), -1)
        canvas_shape = tuple(int(n) for n in geometry.upsampled_scan_gpts)
        layout = _layout(canvas_shape, sampling, fov, reach, rotation, margin)
        if fft_sizes:
            layout["per_fov"] = np.asarray([smooth_fft_size(int(v)) for v in layout["per_fov"]])
            if layout["placement"] == "patch":
                layout["multiples"] = np.asarray(
                    [smooth_fft_size(int(v)) for v in layout["multiples"]])
        layout = _finish_layout(layout, fov, positions_A)
        multiples, per_fov, step = layout["multiples"], layout["per_fov"], layout["step"]
        inner = layout["shape"]

        # the Gram on the canvas's own lattice, from the probe refined `multiples` times
        canvas_lags = tuple(int(n) // 2 // int(m) + 1 for n, m in zip(inner, multiples))
        gram_probe = refined(multiples)
        ops = _operands(gram_probe, gpts, multiples, inner, canvas_lags, device, cdtype)
        gram = _gram(gram_probe, ops, offsets, gpts, multiples, inner, dk, device, rdtype, cdtype)
        del ops, gram_probe

        if layout["placement"] == "lattice":
            sub, grid_shape, lags, probe = multiples, inner, canvas_lags, refined(multiples)
        else:
            sub, grid_shape = fields, tuple(int(v) for v in fields * per_fov)
            # a correlation vanishes beyond 2·reach lags, and a sub-lattice adds under one
            lags = tuple(int(v) for v in np.minimum(per_fov // 2 + 1, 2 * reach + 2))
            probe = patch_probe
        ops = _operands(probe, gpts, sub, grid_shape, lags, device, cdtype)
        padded = ops.padded
        bf_rows = offsets[:, 0] % padded[0]
        bf_cols = offsets[:, 1] % padded[1]
        bf_flat = torch.as_tensor(bf_rows * padded[1] + bf_cols, device=device)
        psi = ops.psi
        bf_weights = float(psi.abs().square().reshape(-1)[bf_flat].sum())
        conj_a_hat = torch.fft.fft2(ops.minus).conj()
        b_hat = torch.fft.fft2(ops.plus)
        q_rows = _dev(np.fft.fftfreq(int(grid_shape[0]), step[0]), device, rdtype)
        q_cols = _dev(np.fft.fftfreq(int(grid_shape[1]), step[1]), device, rdtype)

        positions = layout["positions"]
        if layout["placement"] == "lattice":
            place = (_dev(positions * step, device, rdtype),)
            offs = None
        else:
            base = np.round(positions)
            fraction = _dev((positions - base) * step, device, rdtype)
            place = (torch.as_tensor(base.astype(np.int64), device=device), fraction)
            offs = (torch.as_tensor(signed_index(int(grid_shape[0])), device=device),
                    torch.as_tensor(signed_index(int(grid_shape[1])), device=device))

        # the one-column solve, −i·b / √(Gram·W), as a multiplier on the numerator
        peak = float(gram.max())
        live = gram > OVERLAP_FLOOR**2 * peak
        safe = torch.where(live, gram, torch.ones_like(gram))
        operator = torch.where(live, torch.rsqrt(safe * bf_weights), torch.zeros_like(gram))
        operator = operator.to(cdtype) * (-1j)
        numerator_mask = None
        if layout["placement"] == "patch":
            numerator_mask = gram > _PATCH_FLOOR * peak

        reader = _CanvasReader(inner, step, layout["origin"], layout["rotation"], canvas_shape,
                               sampling, device, rdtype, cdtype)
        sublattice_points = len(ops.minus) * padded[0] * padded[1]
        grid_points = int(grid_shape[0]) * int(grid_shape[1])
        batch = max(1, min(_POINTS_PER_BATCH * 4 // sublattice_points,
                           _POINTS_PER_BATCH * (8 if layout["placement"] == "lattice" else 2)
                           // grid_points))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        return cls(
            key=geometry_key(geometry), device=device, rdtype=rdtype, cdtype=cdtype,
            placement=layout["placement"], multiples=tuple(int(v) for v in multiples),
            per_fov=tuple(int(v) for v in per_fov), inner_shape=tuple(inner),
            grid_shape=tuple(int(v) for v in grid_shape), canvas_shape=canvas_shape,
            n_bf=int(offsets.shape[0]), bf_weights=bf_weights, gram=gram,
            build_s=time.perf_counter() - t0, _ops=ops, _bf_flat=bf_flat,
            _conj_a_hat=conj_a_hat, _b_hat=b_hat, _q=(q_rows, q_cols), _place=place,
            _offsets=offs, _operator=operator, _numerator_mask=numerator_mask, _reader=reader,
            _scale=float(np.prod(sampling) / np.prod(step)), _batch=batch)

    # ── the numerator ──

    def buffer(self) -> torch.Tensor:
        """An empty numerator: the real-space canvas as (pixels, 2) real/imaginary pairs (patch),
        or the numerator's spectrum itself (lattice)."""
        if self.placement == "patch":
            n = self.inner_shape[0] * self.inner_shape[1]
            return torch.zeros((n, 2), dtype=self.rdtype, device=self.device)
        return torch.zeros(self.inner_shape, dtype=self.cdtype, device=self.device)

    def _contract(self, values: torch.Tensor) -> torch.Tensor:
        """Σ_k c_k*(q) V_R(k) on the patch (or canvas) grid, (B, *grid), for values (B, M)."""
        ops = self._ops
        b = values.shape[0]
        p0, p1 = ops.padded
        frames = torch.zeros((b, p0 * p1), dtype=self.cdtype, device=self.device)
        frames[:, self._bf_flat] = values.to(self.cdtype)
        frames = frames.view(b, p0, p1)
        psi = ops.psi
        f_hat = torch.fft.fft2(frames * psi)
        # B(q) = Σ_k ψ(k + q) ψ*(k) V(k) reads the spectrum at −ν
        g_hat = _reflect(torch.fft.fft2(frames * psi.conj()))
        spectra = self._conj_a_hat[:, None] * f_hat[None] - self._b_hat[:, None] * g_hat[None]
        return ops.to_grid(torch.fft.ifft2(spectra))

    def _ramps(self, frequencies: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2j * math.pi * (offsets[:, None] * frequencies[None, :]).to(self.cdtype))

    def deposit(self, buffer: torch.Tensor, values: torch.Tensor, pos: torch.Tensor) -> None:
        """Add the frames at scan positions `pos` (B,), with normalized bright-field values
        `values` (B, M), to `buffer`."""
        values = values.to(self.device, self.rdtype)
        pos = pos.to(self.device, torch.long)
        q_rows, q_cols = self._q
        for lo in range(0, values.shape[0], self._batch):
            v, p = values[lo:lo + self._batch], pos[lo:lo + self._batch]
            if self.placement == "lattice":
                where = self._place[0][p]
                placed = self._contract(v) * self._ramps(q_rows, where[:, 0])[:, :, None]
                buffer += (placed * self._ramps(q_cols, where[:, 1])[:, None, :]).sum(0)
                continue
            base, fraction = self._place[0][p], self._place[1][p]
            # the sub-pixel part of each position, as a phase on the patch's own lattice
            patch = self._contract(v) * self._ramps(q_rows, fraction[:, 0])[:, :, None]
            patch = torch.fft.ifft2(patch * self._ramps(q_cols, fraction[:, 1])[:, None, :])
            i0, i1 = self.inner_shape
            rows = (base[:, 0:1] + self._offsets[0][None, :]) % i0
            cols = (base[:, 1:2] + self._offsets[1][None, :]) % i1
            flat = rows[:, :, None] * i1 + cols[:, None, :]
            buffer.index_add_(0, flat.reshape(-1), torch.view_as_real(patch).reshape(-1, 2))

    def numerator(self, buffer: torch.Tensor) -> torch.Tensor:
        """b(q) on the canvas lattice, from a buffer."""
        if self.placement == "lattice":
            return buffer
        canvas = torch.view_as_complex(buffer.view(*self.inner_shape, 2).contiguous())
        return torch.where(self._numerator_mask, torch.fft.fft2(canvas), 0)

    def image(self, buffer: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        """The OBF phase image, (Ny·U, Nx·U) on the device, zero-mean, optionally multiplied in
        Fourier space by `base` (the parallax readout's Butterworth / Nyquist filters)."""
        obj = torch.fft.ifft2(self.numerator(buffer) * self._operator).real
        # the inverse transform divides by the pixel count, so the same coefficients give an
        # image proportional to the pixel area; the requested canvas holds only part of the inner
        # one, whose mean is zero, so its own mean is set to zero as the solve sets every other's
        obj = self._reader(obj) * self._scale
        obj = obj - obj.mean()
        if base is not None:
            obj = torch.fft.ifft2(torch.fft.fft2(obj) * base).real
        return obj

    def to(self, device) -> "ObfSetup":
        """The same setup on another device (MPS: built on the CPU, then moved)."""
        device = torch.device(device)

        def move(t):
            return None if t is None else t.to(device)

        ops = self._ops
        ops = _Operands(padded=ops.padded, sub=ops.sub, minus=move(ops.minus), plus=move(ops.plus),
                        row_lookup=move(ops.row_lookup), col_lookup=move(ops.col_lookup),
                        kept=move(ops.kept))
        return dataclasses.replace(
            self, device=device, gram=move(self.gram), _ops=ops, _bf_flat=move(self._bf_flat),
            _conj_a_hat=move(self._conj_a_hat), _b_hat=move(self._b_hat),
            _q=tuple(move(t) for t in self._q), _place=tuple(move(t) for t in self._place),
            _offsets=None if self._offsets is None else tuple(move(t) for t in self._offsets),
            _operator=move(self._operator), _numerator_mask=move(self._numerator_mask),
            _reader=self._reader.to(device))

    def nbytes(self) -> int:
        """Device memory the setup holds (roughly)."""
        tensors = [self.gram, self._ops.minus, self._ops.plus, self._conj_a_hat, self._b_hat,
                   self._operator, *self._place]
        return int(sum(t.numel() * t.element_size() for t in tensors))


def reconstruct_once(store: torch.Tensor, visits: torch.Tensor, kind: str, geometry, *,
                     setup: ObfSetup | None = None, subtract_frame_mean: bool = False,
                     device=None, dtype=None, chunk: int = 4096,
                     base: torch.Tensor | None = None) -> torch.Tensor:
    """OBF of every position a `latest` store holds (`visits > 0`), in one call: the oracle of
    the streaming paths, and the offline entry point."""
    if setup is None:
        setup = ObfSetup.build(geometry, device=device if device is not None else store.device,
                               dtype=dtype)
    held = torch.nonzero(visits > 0).squeeze(1).to(store.device)
    mean = bright_field_mean(store, kind, held)
    buffer = setup.buffer()
    for pos in held.split(chunk):
        values = normalize(_raw_float(store[pos], kind), mean, subtract_frame_mean)
        setup.deposit(buffer, values.to(setup.device), pos.to(setup.device))
    return setup.image(buffer, base)
