"""B5: what --subtract-frame-mean removes (repo environment; after run_variants.sh and figures.py).
The frame mean is each position's mean over its bright-field pixels: the incoherent bright-field
intensity there (mass-thickness and diffraction contrast), blurred by the defocused probe
(diameter 2·α·|C10|).

    python frame_mean.py"""
import argparse
import json
import sys

import matplotlib
import numpy as np

from paths import ANALYSIS as A, RECON as R, RECORDINGS, REPO, STEP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(REPO))
import tools.calibrate_parallax as cp  # noqa: E402
from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from realtime_stem.processing.parallax.config import config_from_args  # noqa: E402
from realtime_stem.processing.parallax.engine import preprocess_geometry  # noqa: E402

U = 2
KW = dict(cmap="gray", interpolation="antialiased", interpolation_stage="data")
drift = json.load(open(A / "B_summary.json"))["binned_unbinned"]["drift_px"]
LINE = {"binned": (182, 0, 120), "unbinned": (182 + round(drift[0]), 0, 120)}   # row, col0, col1 (×1 px)

def frame_mean(mode):
    algo = ParallaxAlgorithm(); p = argparse.ArgumentParser(); algo.add_cli_args(p)
    cfg = config_from_args(p.parse_args(algo.preprocess_argv(open(A / f"cal_{mode}.flags").read().split())))
    geo = preprocess_geometry(cfg.detector_shape, cfg.scan_shape, cfg.scan_sampling, cfg.energy, cfg.semiangle_cutoff,
                              angular_sampling=cfg.angular_sampling, aberration_coefs=cfg.aberration_coefs,
                              rotation_angle=cfg.rotation_angle, probe_center=cfg.probe_center, flip_cols=cfg.flip_cols)
    img = np.zeros(65536)
    for ids, frames in cp.file_batches(RECORDINGS[mode])(1):
        img[ids] = frames.reshape(len(frames), -1)[:, geo.bf_flat_inds].mean(1)
    return img.reshape(256, 256), cfg

def radial(a, step):
    a = a - a.mean(); n = a.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fft2(a * w)) ** 2
    q = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]) / step
    bins = np.linspace(0, 0.5 / step, 120)
    i = np.digitize(q.ravel(), bins)
    return 0.5 * (bins[1:] + bins[:-1]), np.bincount(i, p.ravel(), len(bins) + 1)[1:len(bins)] / np.maximum(np.bincount(i, minlength=len(bins) + 1)[1:len(bins)], 1)

def show(ax, a, title):
    lo, hi = np.percentile(a, [0.5, 99.5]); ax.imshow(a, vmin=lo, vmax=hi, **KW)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

fig, ax = plt.subplots(4, 4, figsize=(18, 18.5))
stats = {}
for g, mode in enumerate(("binned", "unbinned")):
    fm, cfg = frame_mean(mode)
    alpha, c10 = cfg.semiangle_cutoff * 1e-3, cfg.aberration_coefs["C10"]
    q_probe = 1 / (2 * alpha * abs(c10))
    kept = np.load(R / f"{mode}_U{U}/parallax_corrected.npy"); sub = np.load(R / f"{mode}_U{U}_sub/parallax_corrected.npy")
    pk = np.load(R / f"{mode}_U{U}/parallax_phase_flipped.npy"); ps = np.load(R / f"{mode}_U{U}_sub/parallax_phase_flipped.npy")
    diff = kept - sub
    r0, c0, c1 = LINE[mode]
    top, bot = ax[2 * g], ax[2 * g + 1]
    show(top[0], fm, f"{mode}: frame mean (mean over the BF disk per position), ×1")
    top[0].plot([c0, c1], [r0, r0], color="tab:orange", lw=1)
    show(top[1], kept, f"parallax ×{U}, frame mean kept (default)")
    show(top[2], sub, f"parallax ×{U}, --subtract-frame-mean")
    show(top[3], diff, "kept − subtracted (what the subtraction removes)")
    show(bot[0], pk, f"phase-flipped ×{U}, kept")
    show(bot[1], ps, f"phase-flipped ×{U}, --subtract-frame-mean")
    for a, lab in ((kept, "kept"), (sub, "subtracted"), (diff, "difference")):
        q, pw = radial(a, STEP / U)
        bot[2].loglog(q[1:], pw[1:], label=lab)
    bot[2].axvline(q_probe, color="gray", ls=":", label=f"1/(2α|C10|) = 1/{1 / q_probe:.0f} Å")
    bot[2].set_xlabel("q (1/Å)"); bot[2].set_ylabel("radial power (parallax)"); bot[2].legend(fontsize=8)
    bot[2].set_title("power spectra, parallax ×2", fontsize=9)
    def smooth(a, sigma_px):
        fy = np.fft.fftfreq(a.shape[0])[:, None]; fx = np.fft.fftfreq(a.shape[1])[None, :]
        return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * np.pi**2 * sigma_px**2 * (fy**2 + fx**2))).real
    sig = 6.0                                                    # Å
    x = np.arange(c0 * U, c1 * U) / U * STEP / 10
    for a, lab in ((kept, "parallax, kept"), (sub, "parallax, subtracted")):
        prof = smooth(a, sig / (STEP / U))[r0 * U, c0 * U:c1 * U]
        bot[3].plot(x, prof / kept.std(), label=lab)
    bot[3].set_xlabel("x (nm) along the orange line"); bot[3].set_ylabel("parallax / rms of 'kept'")
    tw = bot[3].twinx()
    tw.plot(np.arange(c0, c1) * STEP / 10, smooth(fm, sig / STEP)[r0, c0:c1], color="k", lw=0.9,
            alpha=0.7, label="frame mean")
    tw.set_ylabel("frame mean (counts per BF px)")
    h1, l1 = bot[3].get_legend_handles_labels(); h2, l2 = tw.get_legend_handles_labels()
    bot[3].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    bot[3].set_title(f"profile across the ring particle, Gaussian σ = {sig:g} Å", fontsize=9)
    qq, pkp = radial(kept, STEP / U); _, psp = radial(sub, STEP / U)
    low = qq < q_probe
    stats[mode] = {"std_sub_over_kept": float(sub.std() / kept.std()),
                   "low_q_power_sub_over_kept": float(psp[low].sum() / pkp[low].sum()),
                   "q_probe": q_probe}
fig.suptitle("Gd2O3: what --subtract-frame-mean removes (whole 50.6 nm field)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(A / "B5_frame_mean.png", dpi=170)
print(json.dumps(stats, indent=1))
