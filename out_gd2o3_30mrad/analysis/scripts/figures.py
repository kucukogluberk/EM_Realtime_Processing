"""The reconstruction figures: realtime-stem against quantem (B1), the frame mean / DC correction /
upsampling variants (B2), binned against unbinned (B3), and an overview (repo environment):

    python figures.py          # after run_variants.sh (and quantem_recons.py for B1)

Writes full-resolution PNGs and B_summary.json (correlations, drift) to analysis/."""
import json

import matplotlib
import numpy as np

from paths import ANALYSIS as A, QUANTEM as Q, RECON as R, STEP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 200
# filter the data (not the colours) when an image is drawn smaller than its pixel count
IMKW = dict(interpolation="antialiased", interpolation_stage="data")
BIXBYITE = {"211": 4.41, "222": 3.12, "400": 2.70, "440": 1.91, "622": 1.63}   # a = 10.81 Å
CAL = {m: json.load(open(A / f"cal_{m}.json")) for m in ("binned", "unbinned")}


def ours(mode, variant, channel="phase_flipped"):
    f = R / f"{mode}_{variant}" / {"phase_flipped": "parallax_phase_flipped.npy",
                                   "parallax": "parallax_corrected.npy",
                                   "raw": "parallax_recon.npy"}[channel]
    return np.load(f) if f.exists() else None


def quantem(mode, route, kernel, u):
    f = Q / f"{mode}_{route}_{kernel}_U{u}.npy"
    return np.load(f) if f.exists() else None


def norm(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def clim(a, lo=0.5, hi=99.5):
    return np.percentile(a, [lo, hi])


def gauss(a, sigma):
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * np.pi**2 * sigma**2 * (fy**2 + fx**2))).real


def shift_between(a, b):
    """(dy, dx) such that b(r) ≈ a(r − d) (sub-pixel, parabolic), on windowed images."""
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    cc = np.fft.ifft2(np.fft.fft2(norm(b) * w) * np.fft.fft2(norm(a) * w).conj()).real
    cc = np.fft.fftshift(cc)
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    def vertex(lo, c0, hi):
        den = lo - 2 * c0 + hi
        return 0.5 * (lo - hi) / den if den else 0.0
    dy = iy - a.shape[0] // 2 + vertex(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix])
    dx = ix - a.shape[1] // 2 + vertex(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1])
    return dy, dx, cc.max() / (np.sqrt((norm(a) * w) ** 2).sum() * np.sqrt((norm(b) * w) ** 2).sum()) ** 0.5


def fourier_shift(a, d):
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * (fy * d[0] + fx * d[1]))).real


def power(a):
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return np.fft.fftshift(np.abs(np.fft.fft2(norm(a) * w)) ** 2)


def show_power(ax, a, u, qmax=0.75, title=""):
    """log power spectrum out to qmax (1/Å), bixbyite rings dashed."""
    n = a.shape[0]
    dq = 1 / (n * STEP / u)
    half = min(n // 2, int(qmax / dq))
    p = power(a)[n // 2 - half:n // 2 + half, n // 2 - half:n // 2 + half]
    lp = np.log10(p + p.mean() * 1e-3)
    ax.imshow(lp, cmap="magma", vmin=np.percentile(lp, 30), vmax=np.percentile(lp, 99.9),
              extent=[-half * dq, half * dq, half * dq, -half * dq], **IMKW)
    for name, d in BIXBYITE.items():
        if 1 / d < half * dq:
            ax.add_patch(plt.Circle((0, 0), 1 / d, fill=False, ls="--", lw=0.6, color="c"))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def crop(a, u, box):
    (r0, r1), (c0, c1) = box
    return a[r0 * u:r1 * u, c0 * u:c1 * u]


def panel(ax, img, title):
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    else:
        lo, hi = clim(img)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, **IMKW)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


# the zoom: a 64-px (12.6 nm) box around the ring particle, located on the binned image
def find_box(mode, offset=(0, 0)):
    return ((150 + offset[0], 214 + offset[0]), (16 + offset[1], 80 + offset[1]))


def methods(mode, u):
    """(name, image) in display order; quantem's phase has the opposite sign convention for
    some kernels, so each image is shown as is and its correlation with ours is reported."""
    out = [("ours parallax", ours(mode, f"U{u}", "parallax")),
           ("ours phase-flipped", ours(mode, f"U{u}", "phase_flipped"))]
    for route in ("direct", "montage"):
        for k, lab in (("prlx", "prlx"), ("prlx_flip", "prlx+flip"), ("ssb", "SSB"), ("obf", "OBF")):
            out.append((f"{route} {lab}", quantem(mode, route, k, u)))
    return out


def corr(a, b):
    if a is None or b is None or a.shape != b.shape:
        return np.nan
    return float((norm(a) * norm(b)).mean())


def fig_methods(mode, box):
    ups = (1, 2, 4)
    names = [n for n, _ in methods(mode, 1)]
    fig, ax = plt.subplots(2 * len(ups), len(names), figsize=(2.1 * len(names), 4.3 * len(ups)))
    table = {}
    for i, u in enumerate(ups):
        ms = methods(mode, u)
        ref = ms[1][1]
        for j, (n, img) in enumerate(ms):
            r = corr(img, ref)
            table[(u, n)] = r
            panel(ax[2 * i, j], None if img is None else crop(img, u, box),
                  f"{n} ×{u}" + ("" if np.isnan(r) else f"\nr = {r:+.2f} vs ours"))
            if img is None:
                ax[2 * i + 1, j].axis("off")
            else:
                show_power(ax[2 * i + 1, j], img, u, title=f"power ×{u}")
    fig.suptitle(f"Gd2O3 {mode}: realtime-stem and quantem with the calibrated geometry "
                 f"(C10 {CAL[mode]['geometry']['c10']:.0f} Å, θ "
                 f"{CAL[mode]['geometry']['rotation_deg']:.2f}°, flip); zoom 12.6 nm; "
                 f"rings: bixbyite 211/222/400/440/622", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(A / f"B1_methods_{mode}.png", dpi=DPI)
    plt.close(fig)
    return table


def fig_variants(mode, box):
    """Frame mean kept / subtracted, upsampling 1-8, and the static pattern DC correction removes
    (the --no-dc-correct image minus the corrected one), each with its power spectrum on one
    common ±0.75 1/Å axis."""
    ups = (1, 2, 4, 8)
    rows = [("parallax, mean kept (default)", lambda u: ours(mode, f"U{u}", "parallax")),
            ("parallax, --subtract-frame-mean", lambda u: ours(mode, f"U{u}_sub", "parallax")),
            ("removed by DC correction", lambda u: (None if ours(mode, f"U{u}_nodc", "parallax") is None
                                                    else ours(mode, f"U{u}_nodc", "parallax")
                                                    - ours(mode, f"U{u}", "parallax")))]
    fig, ax = plt.subplots(2 * len(rows), len(ups), figsize=(3.3 * len(ups), 6.6 * len(rows)))
    for i, (lab, get) in enumerate(rows):
        for j, u in enumerate(ups):
            img = get(u)
            panel(ax[2 * i, j], None if img is None else crop(img, u, box), f"{lab} ×{u}")
            if img is None or np.allclose(img, img.flat[0], atol=1e-6 * (np.abs(img).max() + 1)):
                ax[2 * i + 1, j].axis("off")
                if img is not None:
                    note = "\n(the comb sits at Nyquist,\nwhich --suppress-nyquist removes)"
                    ax[2 * i + 1, j].text(0.5, 0.5, "constant" + (note if u == 2 else ""),
                                          ha="center", va="center",
                                          transform=ax[2 * i + 1, j].transAxes)
                continue
            show_power(ax[2 * i + 1, j], img, u, title=f"power ×{u}, whole field, ±0.75 1/Å")
            ax[2 * i + 1, j].set_xlim(-0.75, 0.75)
            ax[2 * i + 1, j].set_ylim(0.75, -0.75)
            ax[2 * i + 1, j].set_facecolor("black")
    fig.suptitle(f"Gd2O3 {mode}, realtime-stem parallax channel (zoom 12.6 nm; rings: bixbyite "
                 f"211/222/400/440/622; scan lattice at 1/1.98 Å = 0.506 1/Å)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(A / f"B2_variants_{mode}.png", dpi=DPI)
    plt.close(fig)


def frc(a, b, nbins=60):
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(norm(a) * w), np.fft.fft2(norm(b) * w)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    q = np.hypot(fy, fx)
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(q.ravel(), edges) - 1
    num = np.bincount(idx, (fa * fb.conj()).real.ravel(), nbins + 1)[:nbins]
    da = np.bincount(idx, (np.abs(fa) ** 2).ravel(), nbins + 1)[:nbins]
    db = np.bincount(idx, (np.abs(fb) ** 2).ravel(), nbins + 1)[:nbins]
    return 0.5 * (edges[1:] + edges[:-1]), num / np.sqrt(da * db + 1e-30)


def fig_binned_unbinned():
    a1 = ours("binned", "U1", "parallax")
    b1 = ours("unbinned", "U1", "parallax")
    dy, dx, peak = shift_between(a1, b1)            # unbinned(r) ≈ binned(r − d)
    fig, ax = plt.subplots(2, 3, figsize=(14, 9.4))
    out = {"drift_px": (dy, dx), "drift_A": (dy * STEP, dx * STEP), "peak": peak}
    for u in (1, 2, 4):
        a = ours("binned", f"U{u}")
        b = fourier_shift(ours("unbinned", f"U{u}"), (-dy * u, -dx * u))
        m = int(np.ceil(max(abs(dy), abs(dx)) * u)) + 8 * u
        f, c = frc(a[m:-m, m:-m], b[m:-m, m:-m])
        qa = f / (STEP / u)
        ax[1, 2].plot(qa, c, label=f"×{u}")
        out[f"frc_U{u}"] = (qa.tolist(), c.tolist())
    box = find_box("binned")
    panel(ax[0, 0], crop(ours("binned", "U4"), 4, box), "binned ×4, phase-flipped")
    panel(ax[0, 1], crop(fourier_shift(ours("unbinned", "U4"), (-dy * 4, -dx * 4)), 4, box),
          "unbinned ×4, phase-flipped, registered")
    show_power(ax[1, 0], ours("binned", "U4"), 4, title="binned ×4, whole field")
    show_power(ax[1, 1], ours("unbinned", "U4"), 4, title="unbinned ×4, whole field")
    ax[0, 2].axis("off")
    for name, d in BIXBYITE.items():
        ax[1, 2].axvline(1 / d, color="c", ls=":", lw=0.8)
        ax[1, 2].text(1 / d, 0.95, name, fontsize=7, rotation=90, va="top")
    ax[1, 2].axhline(1 / 7, color="gray", ls="--", lw=0.8)
    ax[1, 2].set_xlabel("q (1/Å)")
    ax[1, 2].set_ylabel("FRC binned vs unbinned (phase-flipped)")
    ax[1, 2].set_xlim(0, 1.0)
    ax[1, 2].legend()
    fig.suptitle(f"binned (14:54) vs unbinned (14:43): drift {dy * STEP:+.1f}, {dx * STEP:+.1f} Å "
                 f"between the scans; zoom 12.6 nm", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(A / "B3_binned_vs_unbinned.png", dpi=DPI)
    plt.close(fig)
    return out


def fig_overview():
    fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
    for i, mode in enumerate(("binned", "unbinned")):
        for j, (v, ch, t) in enumerate([("U1", "parallax", "parallax ×1"),
                                        ("U1", "phase_flipped", "phase-flipped ×1"),
                                        ("U4", "phase_flipped", "phase-flipped ×4")]):
            panel(ax[i, j], ours(mode, v, ch), f"{mode} {t}")
    fig.tight_layout()
    fig.savefig(A / "overview.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig_overview()
    summary = {}
    off = fig_binned_unbinned()
    summary["binned_unbinned"] = {k: v for k, v in off.items() if not k.startswith("frc")}
    for mode in ("binned", "unbinned"):
        box = find_box("binned") if mode == "binned" else find_box(
            "binned", (int(round(off["drift_px"][0])), int(round(off["drift_px"][1]))))
        t = fig_methods(mode, box)
        summary[f"corr_{mode}"] = {f"U{u} {n}": r for (u, n), r in t.items()}
        fig_variants(mode, box)
    json.dump(summary, open(A / "B_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
