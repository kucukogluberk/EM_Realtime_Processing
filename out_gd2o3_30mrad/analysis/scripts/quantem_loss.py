"""quantem's residual loss on the binned recording (quantem's environment), an independent check of
the calibration: along C10 and θ through the calibrated geometry, jointly over C10 and C30, and at
the live, calibrated, higher-order and twin settings, for the prlx and ssb kernels. ~15 min.

    python quantem_loss.py

The finding: both put C10 within one depth of field of the calibration (−300 Å on a 25 Å grid)
and θ at 5° (2° grid), and are flat along the C10–C30 trade-off, so they don't confirm the
apparent C30 of the shift fit (parallax corrects quadratic aberrations only)."""
import json

import numpy as np

import quantem as em

from paths import INPUTS, cal

c = json.load(open(cal("binned")))
g, ho = c["geometry"], c["higher_orders"]
canon = np.load(INPUTS / "binned_canonical.npy", mmap_mode="r").reshape(256, 256, 96, 96)
origin = tuple(json.load(open(INPUTS / "binned_canonical.json"))["origin"])
ds = em.core.datastructures.Dataset4dstem.from_array(
    np.ascontiguousarray(canon, dtype=np.float32),
    sampling=[c["scan_sampling"]] * 2 + [c["angular_sampling"]] * 2, units=["A", "A", "mrad", "mrad"])
dp = em.diffractive_imaging.DirectPtychography.from_dataset4d(
    ds, energy=c["energy"], semiangle_cutoff=c["semiangle"], rotation_angle=g["rotation_deg"],
    aberration_coefs={"C10": g["c10"]}, force_fitted_origin=origin, verbose=False)
c12 = {"C12": g["c12"], "phi12": float(np.deg2rad(g["phi12_deg"]))}


def loss(kernel, ab, rot):
    dp.reconstruct(override_aberration_coefs={**c12, **ab}, override_rotation_angle=rot,
                   deconvolution_kernel=kernel, verbose=False)
    r = dp.residual_loss
    return float(r() if callable(r) else r)


rot0 = g["rotation_deg"]
for kernel in ("prlx", "ssb"):
    c10s = np.arange(-450, -149, 25.0)
    l = [loss(kernel, {"C10": v}, rot0) for v in c10s]
    print(f"{kernel} along C10 (θ {rot0:.2f}°): minimum at {c10s[int(np.argmin(l))]:.0f} Å", flush=True)
    rots = np.arange(-5, 15.1, 2.0)
    l = [loss(kernel, {"C10": g["c10"]}, r) for r in rots]
    print(f"{kernel} along θ: minimum at {rots[int(np.argmin(l))]:.0f}°", flush=True)
    c30s = np.array([-15, -10, -5, -2.5, 0, 2.5, 5, 10]) * 1e4
    grid = np.array([[loss(kernel, {"C10": v, "C30": c3}, rot0) for v in np.arange(-340, -239, 10.0)]
                     for c3 in c30s])
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"{kernel} over (C10, C30): minimum at C30 {c30s[i] / 1e4:+.1f} µm, C10 {-340 + 10 * j} Å; "
          f"best per C30: " + ", ".join(f"{c3 / 1e4:+.1f} µm → {row.min():.5f}"
                                         for c3, row in zip(c30s, grid)), flush=True)
    points = {
        "calibrated (run.py) geometry": ({"C10": g["c10"]}, rot0),
        "with the apparent higher orders": ({"C10": ho["c10"], "C30": ho["c30"], "C21": ho["c21"],
                                             "phi21": np.deg2rad(ho["phi21_deg"]), "C23": ho["c23"],
                                             "phi23": np.deg2rad(ho["phi23_deg"])}, ho["rotation_deg"]),
        "live settings (C10 -300 Å, θ 3.75°, no C12)": ({"C10": -300.0, "C12": 0.0}, 3.75),
        "the twin (−C10, θ + 180°)": ({"C10": -g["c10"], "C12": g["c12"],
                                       "phi12": float(np.deg2rad(g["phi12_deg"] + 90))}, rot0 + 180),
    }
    for name, (ab, rot) in points.items():
        print(f"  {kernel} {name}: {loss(kernel, ab, rot):.5f}", flush=True)
