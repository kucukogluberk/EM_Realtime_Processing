"""Does OBF need the aberrations beyond C10/C12? quantem's correlation-route OBF of the binned scan at
x2 with the calibrated quadratic geometry (as run.py gets it) and with the calibration's
higher-order fit (C30, C21, C23, its own C10/C12), against our phase-flipped parallax, by how far
the bixbyite spots stand out of their ring (quantem's environment) → figures/full/obf_aberrations.png

    $QUANTEM_PY obf_aberrations.py"""
import json
import math

import matplotlib
import numpy as np

from paths import FULL, INPUTS, cal, geometry, recon

import quantem as em
from quantem.diffractive_imaging import DirectPtychographyMontage
from quantem.diffractive_imaging.direct_ptycho_utils import build_vbf_stack_from_dataset4d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

U = 2
c = json.load(open(cal("binned")))
g, ho = geometry("binned"), c["higher_orders"]
canon = np.load(INPUTS / "binned_canonical.npy", mmap_mode="r").reshape(256, 256, 96, 96)
origin = tuple(json.load(open(INPUTS / "binned_canonical.json"))["origin"])
ds = em.core.datastructures.Dataset4dstem.from_array(
    np.ascontiguousarray(canon, dtype=np.float32),
    sampling=[c["scan_sampling"]] * 2 + [c["angular_sampling"]] * 2, units=["A", "A", "mrad", "mrad"])
vbf, mask, rotation = build_vbf_stack_from_dataset4d(
    ds, device="cpu", force_fitted_origin=origin, rotation_angle=g["rotation_deg"],
    aperture=DirectPtychographyMontage._model_aperture_for(c["energy"], None, c["semiangle"], None,
                                                           "cpu"))
del ds
variants = {
    "quadratic (live)": {"C10": g["c10"], "C12": g["c12"], "phi12": math.radians(g["phi12_deg"])},
    "+ C30, C21, C23 (fit)": {
        "C10": ho["c10"], "C12": ho["c12"], "phi12": math.radians(ho["phi12_deg"]), "C30": ho["c30"],
        "C21": ho["c21"], "phi21": math.radians(ho["phi21_deg"]), "C23": ho["c23"],
        "phi23": math.radians(ho["phi23_deg"])},
}
images = {"parallax, phase-flipped": np.load(recon("binned", f"U{U}") / "parallax_phase_flipped.npy")}
for name, ab in variants.items():
    model = DirectPtychographyMontage.from_virtual_bfs(
        vbf_dataset=vbf, bf_mask_dataset=mask, energy=c["energy"], semiangle_cutoff=c["semiangle"],
        rotation_angle=rotation, aberration_coefs=ab, device="mps", verbose=False)
    model.reconstruct(deconvolution_kernel="obf", convolution_mode="correlation",
                      upsampling_factor=U, verbose=False)
    images[f"OBF {name}"] = np.asarray(model.obj, dtype=np.float64)

n, t = 256 * U, 256 * U // 8
win = np.hanning(n - 2 * t)[:, None] * np.hanning(n - 2 * t)[None, :]
q = np.hypot(*np.meshgrid(np.fft.fftfreq(n - 2 * t, c["scan_sampling"] / U),
                          np.fft.fftfreq(n - 2 * t, c["scan_sampling"] / U), indexing="ij"))
rings = {"222": 3.12, "400": 2.70, "440": 1.91, "622": 1.63}
fig, ax = plt.subplots(2, len(images), figsize=(4.2 * len(images), 8.6))
for i, (name, img) in enumerate(images.items()):
    a = np.asarray(img, np.float64)[t:-t, t:-t]
    p = np.abs(np.fft.fft2((a - a.mean()) * win)) ** 2
    stand = []
    for hkl, d in rings.items():
        r = np.abs(q - 1 / d) < 0.025 / d
        top = np.sort(p[r])[-max(3, r.sum() // 500):].mean()
        stand.append(f"{hkl} {top / np.median(p[r]):.0f}x")
    print(f"{name:>34}: spots over their ring's median: " + ", ".join(stand), flush=True)
    lo, hi = np.percentile(a, [0.5, 99.5])
    ax[0, i].imshow(a, cmap="gray", vmin=lo, vmax=hi)
    ax[0, i].set_title(name, fontsize=10)
    nq = 0.5 * U / c["scan_sampling"]                   # Nyquist, 1/Å
    ax[1, i].imshow(np.log10(np.fft.fftshift(p) + 1e-12), cmap="magma",
                    extent=(-nq, nq, nq, -nq), vmin=np.percentile(np.log10(p + 1e-12), 5))
    for d in rings.values():
        ax[1, i].add_patch(plt.Circle((0, 0), 1 / d, fill=False, ls=":", lw=0.6, color="c"))
    ax[1, i].set_title(", ".join(stand), fontsize=8)
    for a_ in ax[:, i]:
        a_.axis("off")
plt.tight_layout()
FULL.mkdir(parents=True, exist_ok=True)
plt.savefig(FULL / "obf_aberrations.png", dpi=110)
print("saved", FULL / "obf_aberrations.png")
