"""The bright-field disk on 2026-10-23 against 2026-10-20 (repo environment): its radius in detector
pixels from the 50% point of the azimuthal profile of the mean pattern (the calibration's area-based
`fit_disk` also counts the dark-field halo, which the Au scatters far more of), the halo, and the
disk's centre and radius per 8×8-position tile (does it wander across the scan?).

    python disk_size.py        # ~1 min → figures/full/disk_vs_oct20.png, analysis/summary.json

Finding: 29.9 px against 28.15 px (+6.2%), the same in every tile (no wander, no halo effect), so
α × the effective camera length at the ARINA changed between the sessions."""
import json
import sys

import matplotlib
import numpy as np

from paths import ANALYSIS, FULL, RECORDINGS, REPO, oct20_recording

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(REPO))
from tools.calibrate_parallax import file_batches, fit_disk  # noqa: E402

HOT = [(61, 90), (75, 88)]                       # the ARINA's two hot pixels (binned), outside


def frames_of(path, every=1):
    for _, f in file_batches(path)(every):
        yield f.astype(np.float32)


def mean_pattern(path, every=8):
    tot, n = 0.0, 0
    for f in frames_of(path, every):
        tot, n = tot + f.sum(0), n + len(f)
    m = tot / n
    for r, c in HOT:
        m[r, c] = np.median(m)
    return m


def profile(m, centre, step=0.25):
    yy, xx = np.indices(m.shape)
    r = np.hypot(yy - centre[0], xx - centre[1])
    bins = np.arange(0, 46, step)
    idx = np.digitize(r.ravel(), bins)
    prof = np.bincount(idx, m.ravel(), len(bins) + 1)[1:len(bins)] / np.maximum(
        np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)], 1)
    rc = 0.5 * (bins[1:] + bins[:-1])
    plateau = np.median(prof[(rc > 5) & (rc < 20)])
    bg = prof[(rc > 38) & (rc < 44)].min()
    return rc, (prof - bg) / (plateau - bg)


def edge50(rc, n):
    k = np.flatnonzero((n[:-1] >= 0.5) & (n[1:] < 0.5) & (rc[:-1] > 15))[0]
    return float(rc[k] + (n[k] - 0.5) / (n[k] - n[k + 1]) * (rc[k + 1] - rc[k]))


def per_tile(path, scan=128, tile=8):
    nt = scan // tile
    tiles = np.zeros((nt, nt, 96, 96))
    for ids, f in file_batches(path)(1):
        iy, ix = np.divmod(ids, scan)
        np.add.at(tiles, (iy // tile, ix // tile), f.astype(np.float64))
    tiles /= tile * tile
    centre, radius = np.zeros((nt, nt, 2)), np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            d = fit_disk(tiles[i, j])
            centre[i, j] = d.center
            radius[i, j] = edge50(*profile(tiles[i, j], d.center))
    return centre, radius


pats = {"Gd2O3 binned (2026-10-20)": mean_pattern(oct20_recording("binned"))}
pats.update({f"grating {rid} (2026-10-23)": mean_pattern(p) for rid, p in RECORDINGS.items()})
out = {}
fig = plt.figure(figsize=(18, 10), dpi=110)
gs = fig.add_gridspec(2, 4)
ax_p = fig.add_subplot(gs[1, :2])
for j, (name, m) in enumerate(pats.items()):
    d = fit_disk(m)
    rc, n = profile(m, d.center)
    r50 = edge50(rc, n)
    out[name] = {"centre_px": [round(v, 2) for v in d.center], "radius_50_px": round(r50, 2),
                 "radius_fit_disk_px": round(d.radius, 2),
                 "halo_2_8_px_out": round(float(n[(rc > r50 + 2) & (rc < r50 + 8)].mean()), 3)}
    a = fig.add_subplot(gs[0, j])
    a.imshow(np.log10(np.clip(m, 1e-3, None)), cmap="magma")
    a.add_patch(plt.Circle(d.center[::-1], r50, fill=False, color="c", lw=0.8))
    a.set_title(f"{name}\nmean pattern (log), 50% edge r = {r50:.2f} px", fontsize=9)
    a.set_xticks([]), a.set_yticks([])
    ax_p.plot(rc, n, label=f"{name}: 50% at {r50:.2f} px (fit_disk {d.radius:.2f})")
ax_p.axhline(0.5, color="k", lw=0.5)
ax_p.set_xlim(0, 46), ax_p.set_xlabel("radius (binned px)"), ax_p.set_ylabel("(I − bg)/(plateau − bg)")
ax_p.legend(fontsize=8), ax_p.set_title("azimuthal profiles about each disk centre")

centre, radius = per_tile(RECORDINGS["centered"])
out["per_tile_centered"] = {
    "radius_50_px": {"median": round(float(np.median(radius)), 2),
                     "p5": round(float(np.percentile(radius, 5)), 2),
                     "p95": round(float(np.percentile(radius, 95)), 2)},
    "centre_spread_px": [round(float(np.ptp(centre[..., k])), 2) for k in (0, 1)]}
for k, (lab, a) in enumerate((("disk centre row per 8×8 tile (px)", centre[..., 0]),
                              ("50% radius per 8×8 tile (px)", radius))):
    ax = fig.add_subplot(gs[1, 2 + k])
    im = ax.imshow(a, cmap="RdBu_r" if k == 0 else "viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"grating centered: {lab}", fontsize=9)
ref = out["Gd2O3 binned (2026-10-20)"]["radius_50_px"]
fig.suptitle("The bright-field disk: 2026-10-23 against 2026-10-20 (same detector, binning, energy). "
             f"Grating / Gd₂O₃ = {np.mean([v['radius_50_px'] for k, v in out.items() if 'grating' in k]) / ref:.3f}",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
FULL.mkdir(parents=True, exist_ok=True)
fig.savefig(FULL / "disk_vs_oct20.png")
summary_path = ANALYSIS / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
summary["disk"] = out
summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
