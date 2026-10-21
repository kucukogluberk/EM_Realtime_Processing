"""quantem reconstructions of the recordings with the calibrated geometry (quantem's environment:
../quantem-upstream/.venv/bin/python): DirectPtychography and DirectPtychographyMontage, kernels
prlx without / with the phase flip, ssb and obf, upsampling 1, 2 and 4 → recon/quantem/.

    python quantem_recons.py binned        # ~10 min on an M4 Pro; unbinned ~25 min, ~10 GB RAM

The inputs come from export_for_quantem.py (column-flipped to quantem's canonical layout); the
fitted diffraction origin is forced to the measured disk centre, so both codes use one geometry.
Existing results are skipped, so an interrupted run resumes."""
import json
import sys
import time

import numpy as np
import torch

import quantem as em
from quantem.diffractive_imaging import DirectPtychography, DirectPtychographyMontage

from paths import INPUTS, QUANTEM, cal

mode = sys.argv[1]
QUANTEM.mkdir(parents=True, exist_ok=True)
c = json.load(open(cal(mode)))
g = c["geometry"]
size = 96 if mode == "binned" else 128
canon = np.load(INPUTS / f"{mode}_canonical.npy", mmap_mode="r").reshape(256, 256, size, size)
origin = tuple(json.load(open(INPUTS / f"{mode}_canonical.json"))["origin"])
t = time.time()
ds = em.core.datastructures.Dataset4dstem.from_array(
    np.ascontiguousarray(canon, dtype=np.float32),
    sampling=[c["scan_sampling"]] * 2 + [c["angular_sampling"]] * 2, units=["A", "A", "mrad", "mrad"])
ab = {"C10": g["c10"], "C12": g["c12"], "phi12": float(np.deg2rad(g["phi12_deg"]))}
common = dict(energy=c["energy"], semiangle_cutoff=c["semiangle"], rotation_angle=g["rotation_deg"],
              aberration_coefs=ab, force_fitted_origin=origin, verbose=False)
routes = {"direct": DirectPtychography.from_dataset4d(ds, **common),
          "montage": DirectPtychographyMontage.from_dataset4d(ds, **common)}
print(f"[{mode}] built both routes in {time.time() - t:.0f}s", flush=True)
del ds
kernels = [("prlx", dict(deconvolution_kernel="prlx", parallax_flip_phase=False)),
           ("prlx_flip", dict(deconvolution_kernel="prlx", parallax_flip_phase=True)),
           ("ssb", dict(deconvolution_kernel="ssb")),
           ("obf", dict(deconvolution_kernel="obf"))]
for u in (1, 2, 4):
    for route, obj in routes.items():
        for name, kw in kernels:
            f = QUANTEM / f"{mode}_{route}_{name}_U{u}.npy"
            if f.exists():
                continue
            t = time.time()
            res = obj.reconstruct(upsampling_factor=u, verbose=False, **kw).obj
            arr = res.detach().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res)
            np.save(f, (np.angle(arr) if np.iscomplexobj(arr) else arr).astype(np.float32))
            print(f"[{mode}] {route} {name} ×{u}: {arr.shape} in {time.time() - t:.1f}s", flush=True)
