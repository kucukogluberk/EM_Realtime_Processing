"""The scan step from the Au lattice (repo environment): the live snapshots exported from the viewer
(×4 phase-flipped, 512² = the 128² scan; after the A1 tuning, so their fringes are strong) show Au
reflections; their periods in output pixels, indexed as 111 and 200 (period ratio 1.155), give
the scan step independently of the detector and of C10.

    python scan_step.py        # → figures/full/scan_step.png, analysis/summary.json

Finding: 7.04 and 6.07 output px (ratio 1.160): 1.341 Å per scan step, 4% under the nominal
17.9 nm / 128 = 1.398 Å. It assumes the snapshots were taken at the recordings' field of view."""
import json

import matplotlib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from paths import ANALYSIS, AU, FULL, SNAPSHOTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

U = 4                                              # the snapshots' upsampling


def smooth(a, s):
    fy, fx = np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * np.pi**2 * s**2 * (fy**2 + fx**2))).real


def spots(img, fmin=0.12, fmax=0.26, n=6):
    """The strongest spots (period in output px, azimuth) in the band, one per Friedel pair."""
    N = img.shape[0]
    w = np.outer(np.hanning(N), np.hanning(N))
    P = np.fft.fftshift(np.abs(np.fft.fft2((img - img.mean()) * w)) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(N))
    yy, xx = np.indices(P.shape)
    fr = np.hypot(f[yy], f[xx])
    S = smooth(np.log10(P + P.mean() * 1e-6), 1.0)
    band = (fr > fmin) & (fr < fmax)
    peak = S >= sliding_window_view(np.pad(S, 4, mode="edge"), (9, 9)).max(axis=(2, 3))
    cand = np.argwhere(peak & band & ((f[xx] > 0) | ((f[xx] == 0) & (f[yy] > 0))))
    z = (S[tuple(cand.T)] - np.median(S[band])) / np.std(S[band])
    out = []
    for y, x in cand[np.argsort(z)[::-1][:n]]:
        win = P[y - 2:y + 3, x - 2:x + 3]
        wy, wx = np.indices(win.shape)
        cy, cx = y - 2 + (win * wy).sum() / win.sum(), x - 2 + (win * wx).sum() / win.sum()
        fy, fx = (cy - N // 2) / N, (cx - N // 2) / N
        out.append((1 / np.hypot(fy, fx), float(np.degrees(np.arctan2(fy, fx)))))
    return P, out


fig, ax = plt.subplots(2, len(SNAPSHOTS), figsize=(7 * len(SNAPSHOTS), 13), dpi=110, squeeze=False)
result = {}
for j, path in enumerate(SNAPSHOTS):
    img = np.load(path).astype(np.float64)
    P, found = spots(img)
    p111 = max(p for p, _ in found[:2])            # the two strongest: 111 (longer) and 200
    p200 = min(p for p, _ in found[:2])
    step = (AU["111"] / p111 + AU["200"] / p200) / 2 * U
    result[path.name] = {"periods_out_px": [round(p, 3) for p, _ in found],
                         "111_px": round(p111, 3), "200_px": round(p200, 3),
                         "ratio": round(p111 / p200, 3), "scan_step_A": round(step, 4)}
    lo, hi = np.percentile(img, [0.5, 99.5])
    ax[0, j].imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased")
    ax[0, j].set_title(f"{path.name}: live ×4 phase-flipped", fontsize=10)
    N = img.shape[0]
    L = np.log10(P + P.mean() * 1e-6)
    v0, v1 = np.percentile(L, [30, 99.97])
    ax[1, j].imshow(L, cmap="inferno", vmin=v0, vmax=v1, interpolation="antialiased")
    for (hkl, d), c in zip(AU.items(), ("c", "lime", "deepskyblue")):
        ax[1, j].add_patch(plt.Circle((N / 2, N / 2), N * (step / U) / d, fill=False, color=c,
                                      lw=0.6, ls="--"))
    ax[1, j].set_xlim(N / 2 - 160, N / 2 + 160), ax[1, j].set_ylim(N / 2 + 160, N / 2 - 160)
    ax[1, j].set_title(f"111 at {p111:.2f} px, 200 at {p200:.2f} px (ratio {p111 / p200:.3f}, "
                       f"Au 1.155): step {step:.3f} Å", fontsize=10)
    for a in ax[:, j]:
        a.set_xticks([]), a.set_yticks([])
steps = [v["scan_step_A"] for v in result.values()]
result["scan_step_A"] = round(float(np.mean(steps)), 3)
fig.suptitle(f"The scan step from the Au lattice: {result['scan_step_A']:.3f} Å (nominal 17.9 nm / "
             f"128 = 1.398 Å); rings at the measured step", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
FULL.mkdir(parents=True, exist_ok=True)
fig.savefig(FULL / "scan_step.png")
summary_path = ANALYSIS / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary["scan_step"] = result
summary_path.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(result, indent=1))
