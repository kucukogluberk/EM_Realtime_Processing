"""A cross-check of angular sampling × scan step in both sessions (repo environment): the CTF zeros
of the support-focus reconstruction (recon/<id>/U4_fit, the raw sum) against those the
calibration predicts.

At its own focus the support's tcBF image transfers as sin χ(q), χ = πλC10q² (+ C12), with no
detector term. Averaging the power along contours of the calibration's χ (C12 included, so
astigmatism doesn't smear the zeros), the zeros fall at χ/π = n·r, where
r = (Δα·step)_true / (Δα·step)_assumed. r = 1 when the calibration's angular sampling and scan
step are right (the shift fit measures C10·Δα/step; the zeros C10/step²).

    python ctf_zeros.py        # → figures/full/ctf_zeros.png, analysis/summary.json

Finding: Oct 20 r = 0.95–0.97, Oct 23 r = 1.03–1.04 (n = 1–3), with a ~3% systematic from the
fitted C10 itself, so neither session is clearly miscalibrated by this test alone."""
import json
import sys

import matplotlib
import numpy as np

from paths import ANALYSIS, FULL, OCT20, REPO, SESSION, cal, recon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(REPO))
from realtime_stem.processing.parallax.engine import _electron_wavelength  # noqa: E402
from realtime_stem.processing.parallax.readout import quadratic_aberration_surface  # noqa: E402

U = 4
SETS = [(OCT20, "binned"), (OCT20, "unbinned"), (SESSION, "centered"), (SESSION, "not_centered")]


def zeros(session, rid, nmax=3):
    c = cal(rid, session)
    g = c["geometry"]
    lam = _electron_wavelength(c["energy"])
    img = np.load(recon(rid, "U4_fit", session) / "parallax_recon.npy").astype(np.float64)
    N = img.shape[0]
    step = c["scan_sampling"] / U
    w = np.outer(np.hanning(N), np.hanning(N))
    P = np.abs(np.fft.fft2((img - img.mean()) * w)) ** 2
    fy = np.fft.fftfreq(N, step)[:, None] * np.ones((1, N))
    fx = np.ones((N, 1)) * np.fft.fftfreq(N, step)[None, :]
    q, th = np.hypot(fy, fx), np.arctan2(fx, fy)        # the readout's convention
    chi = np.abs(quadratic_aberration_surface(q * lam, th, lam, {
        "C10": g["c10"], "C12": g["c12"], "phi12": np.radians(g["phi12_deg"])})) / np.pi
    k = np.fft.fftfreq(N) * N                           # mask the scan lattice's comb
    near = lambda a: np.abs(((a + N / U / 2) % (N / U)) - N / U / 2) <= 2   # noqa: E731
    ok = ~(near(k[:, None] * np.ones((1, N))) & near(np.ones((N, 1)) * k[None, :])) & (q > 0.05)
    bins = np.arange(0.2, nmax + 0.6, 0.02)
    idx = np.digitize(chi[ok], bins)
    prof = np.bincount(idx, P[ok], len(bins) + 1)[1:len(bins)] / np.maximum(
        np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)], 1)
    u = 0.5 * (bins[1:] + bins[:-1])
    sm = np.convolve(np.log10(prof), np.ones(5) / 5, mode="same")
    found = []
    for n in range(1, nmax + 1):
        win = np.flatnonzero((u > 0.85 * n) & (u < 1.15 * n))
        i = win[np.argmin(sm[win])]
        a, b, _ = np.polyfit(u[i - 3:i + 4], sm[i - 3:i + 4], 2)
        found.append(float(-b / (2 * a)) if a > 0 else float(u[i]))
    r = [f / n for n, f in enumerate(found, 1)]
    return u, sm, found, r, c


fig, ax = plt.subplots(len(SETS), 1, figsize=(13, 3.2 * len(SETS)), dpi=110)
out = {}
for a, (session, rid) in zip(ax, SETS):
    u, sm, found, r, c = zeros(session, rid)
    name = f"{session.name} {rid}"
    out[name] = {"chi_over_pi_at_zeros": [round(v, 3) for v in found], "r": [round(v, 3) for v in r],
                 "r_mean": round(float(np.mean(r)), 3),
                 "assumed": {"angular_sampling": c["angular_sampling"],
                             "scan_sampling": c["scan_sampling"], "c10_fit": round(c["geometry"]["c10"], 1)}}
    a.plot(u, sm, lw=0.8)
    for n, f in enumerate(found, 1):
        a.axvline(n, color="k", lw=0.4, ls=":")
        a.axvline(f, color="r", lw=0.8)
    a.set_title(f"{name}: zeros at χ/π = {', '.join(f'{v:.3f}' for v in found)} → r = "
                f"{np.mean(r):.3f} (assumed {c['angular_sampling']} mrad/px × {c['scan_sampling']} Å)",
                fontsize=10)
    a.set_ylabel("log power")
ax[-1].set_xlabel("|χ| / π of the calibration (dotted: where the zeros should fall)")
fig.tight_layout()
FULL.mkdir(parents=True, exist_ok=True)
fig.savefig(FULL / "ctf_zeros.png")
summary_path = ANALYSIS / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary["ctf_zeros"] = out
summary_path.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(out, indent=1))
