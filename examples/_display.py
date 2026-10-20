"""Small display helpers shared by the example consumers (numpy only, no matplotlib)."""

from __future__ import annotations

import numpy as np


def high_pass(img, sigma_px: float):
    """`img` minus a Gaussian low-pass (sigma in real-space pixels). sigma 0 = identity.

    Pure-numpy FFT (no scipy dependency), NaN-safe. Useful for phase-like reconstructions where a
    slow background swamps the contrast; leave it off (0) for parallax bright-field.
    """
    a = np.asarray(img, dtype=np.float64)
    if sigma_px <= 0:
        return a.copy()
    if not np.isfinite(a).all():
        finite = a[np.isfinite(a)]
        fill = float(np.median(finite)) if finite.size else 0.0
        a = np.where(np.isfinite(a), a, fill)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    gauss = np.exp(-2.0 * np.pi**2 * sigma_px**2 * (fy**2 + fx**2))
    lowpass = np.fft.ifft2(np.fft.fft2(a) * gauss).real
    return a - lowpass


def robust_clim(img, lo_pct: float = 0.5, hi_pct: float = 99.5):
    """Display range from percentiles — so a hot pixel doesn't wash out the image."""
    finite = np.asarray(img)[np.isfinite(img)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if not hi > lo:
        lo, hi = lo - 0.5, lo + 0.5
    return float(lo), float(hi)
