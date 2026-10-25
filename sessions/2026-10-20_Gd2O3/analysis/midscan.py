"""The live view mid-scan (repo environment): snapshots at 25/50/75/100% of each recording,
with DC correction on and off, phase-flipped, and with --normalize, as the live viewer sees them.

    python midscan.py"""
import argparse
import sys

import matplotlib
import numpy as np

from paths import FULL, REPO, RECORDINGS, flags

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(REPO))
import tools.calibrate_parallax as cp  # noqa: E402
from realtime_stem.processing.parallax.algorithm import ParallaxAlgorithm  # noqa: E402
from realtime_stem.processing.parallax.config import config_from_args  # noqa: E402
from realtime_stem.processing.parallax.engine import ParallaxAccumulator, preprocess_geometry  # noqa: E402
from realtime_stem.processing.parallax.readout import ReadoutConfig  # noqa: E402

KW = dict(cmap="gray", interpolation="antialiased", interpolation_stage="data")
READ = {"parallax, DC correction on (default)": (ReadoutConfig(), "parallax"),
        "parallax, --no-dc-correct": (ReadoutConfig(dc_correct=False), "parallax"),
        "phase-flipped (default)": (ReadoutConfig(), "phase_flipped"),
        "phase-flipped, --normalize": (ReadoutConfig(normalize=True), "phase_flipped")}
STOPS = (0.25, 0.5, 0.75, 1.0)

for mode, u in (("binned", 4), ("unbinned", 4), ("binned", 1)):
    algo = ParallaxAlgorithm()
    p = argparse.ArgumentParser()
    algo.add_cli_args(p)
    cfg = config_from_args(p.parse_args(algo.preprocess_argv(flags(mode) + ["--upsampling", str(u)])))
    geo = preprocess_geometry(cfg.detector_shape, cfg.scan_shape, cfg.scan_sampling, cfg.energy,
                              cfg.semiangle_cutoff, angular_sampling=cfg.angular_sampling,
                              aberration_coefs=cfg.aberration_coefs, rotation_angle=cfg.rotation_angle,
                              upsampling_factor=u, probe_center=cfg.probe_center,
                              flip_cols=cfg.flip_cols)
    acc = ParallaxAccumulator(geo)
    snaps, done, i = {}, 0, 0
    for ids, frames in cp.file_batches(RECORDINGS[mode], batch=2048)(1):
        acc.process(frames, ids)
        done += len(ids)
        while i < len(STOPS) and done >= STOPS[i] * 65536:
            snaps[STOPS[i]] = {}
            for name, (rc, ch) in READ.items():
                acc.readout_config, acc._readout = rc, None
                snaps[STOPS[i]][name] = acc.channels()[ch].cpu().numpy()
            i += 1
    fig, ax = plt.subplots(len(READ), len(STOPS), figsize=(4.2 * len(STOPS), 4.3 * len(READ)))
    for c, f in enumerate(STOPS):
        rows_done = int(round(f * 256)) * u
        for r, name in enumerate(READ):
            img = snaps[f][name]
            lo, hi = np.percentile(img[:max(rows_done - 12 * u, 8)], [0.5, 99.5])   # scanned part
            ax[r, c].imshow(img, vmin=lo, vmax=hi, **KW)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"{int(f * 100)}% of the scan", fontsize=11)
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=10)
    fig.suptitle(f"Gd2O3 {mode} ×{u}, live view mid-scan (each panel scaled to its scanned part)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FULL.mkdir(parents=True, exist_ok=True)
    fig.savefig(FULL / f"midscan_{mode}_x{u}.png", dpi=200 if u > 1 else 120)
    plt.close(fig)
    print(mode, u, "done")
